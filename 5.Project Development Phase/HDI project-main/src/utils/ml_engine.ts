/**
 * Machine Learning Engine for HDI Prediction
 * Includes custom implementations of:
 * - StandardScaler (Feature Scaling)
 * - Linear Regression (OLS Closed-Form Solver)
 * - Decision Tree Regressor (Variance Reduction Splitter)
 * - Random Forest Regressor (Ensemble of Decision Trees)
 * - Gradient Boosting Regressor (Boosting Tree Regressor)
 * - Model evaluation metrics (R2, MAE, MSE, RMSE)
 * 
 * Written in robust, highly optimized, type-safe TypeScript.
 */

// --- 1. Utilities and Types ---

export interface ModelMetrics {
  r2: number;
  mae: number;
  mse: number;
  rmse: number;
}

export interface TrainingResult {
  modelName: string;
  metrics: ModelMetrics;
  predictions: number[];
  residuals: number[];
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

// Simple Matrix Algebra Helpers
class Matrix {
  static transpose(X: number[][]): number[][] {
    const rows = X.length;
    const cols = X[0].length;
    const result: number[][] = Array.from({ length: cols }, () => new Array(rows).fill(0));
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result[j][i] = X[i][j];
      }
    }
    return result;
  }

  static multiply(A: number[][], B: number[][]): number[][] {
    const rowsA = A.length;
    const colsA = A[0].length;
    const colsB = B[0].length;
    const result: number[][] = Array.from({ length: rowsA }, () => new Array(colsB).fill(0));
    for (let i = 0; i < rowsA; i++) {
      for (let j = 0; j < colsB; j++) {
        let sum = 0;
        for (let k = 0; k < colsA; k++) {
          sum += A[i][k] * B[k][j];
        }
        result[i][j] = sum;
      }
    }
    return result;
  }

  static multiplyMV(A: number[][], v: number[]): number[] {
    const rows = A.length;
    const cols = A[0].length;
    const result = new Array(rows).fill(0);
    for (let i = 0; i < rows; i++) {
      let sum = 0;
      for (let j = 0; j < cols; j++) {
        sum += A[i][j] * v[j];
      }
      result[i] = sum;
    }
    return result;
  }

  // Gaussian elimination with partial pivoting for small square matrices
  static invert(A: number[][]): number[][] {
    const n = A.length;
    // Create augmented matrix [A | I]
    const aug = A.map((row, i) => {
      const augRow = [...row];
      for (let j = 0; j < n; j++) {
        augRow.push(j === i ? 1 : 0);
      }
      return augRow;
    });

    for (let i = 0; i < n; i++) {
      // Find pivot
      let maxRow = i;
      let maxVal = Math.abs(aug[i][i]);
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(aug[k][i]) > maxVal) {
          maxVal = Math.abs(aug[k][i]);
          maxRow = k;
        }
      }

      // Swap rows
      if (maxRow !== i) {
        const temp = aug[i];
        aug[i] = aug[maxRow];
        aug[maxRow] = temp;
      }

      const pivot = aug[i][i];
      if (Math.abs(pivot) < 1e-10) {
        throw new Error("Matrix is singular and cannot be inverted.");
      }

      // Normalize row i
      for (let j = i; j < 2 * n; j++) {
        aug[i][j] /= pivot;
      }

      // Eliminate other rows
      for (let k = 0; k < n; k++) {
        if (k !== i) {
          const factor = aug[k][i];
          for (let j = i; j < 2 * n; j++) {
            aug[k][j] -= factor * aug[i][j];
          }
        }
      }
    }

    // Extract inverted matrix
    return aug.map(row => row.slice(n));
  }
}

// --- 2. Preprocessing & Scaler ---

export class StandardScaler {
  means: number[] = [];
  stds: number[] = [];
  isFitted = false;

  fit(X: number[][]): void {
    const n = X.length;
    const numFeatures = X[0].length;
    this.means = new Array(numFeatures).fill(0);
    this.stds = new Array(numFeatures).fill(0);

    // Compute means
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < numFeatures; j++) {
        this.means[j] += X[i][j];
      }
    }
    for (let j = 0; j < numFeatures; j++) {
      this.means[j] /= n;
    }

    // Compute standard deviations
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < numFeatures; j++) {
        const diff = X[i][j] - this.means[j];
        this.stds[j] += diff * diff;
      }
    }
    for (let j = 0; j < numFeatures; j++) {
      this.stds[j] = Math.sqrt(this.stds[j] / n);
      if (this.stds[j] < 1e-10) {
        this.stds[j] = 1.0; // avoid division by zero
      }
    }

    this.isFitted = true;
  }

  transform(X: number[][]): number[][] {
    if (!this.isFitted) throw new Error("Scaler must be fitted before transforming data.");
    return X.map(row => row.map((val, j) => (val - this.means[j]) / this.stds[j]));
  }

  fitTransform(X: number[][]): number[][] {
    this.fit(X);
    return this.transform(X);
  }

  transformVector(v: number[]): number[] {
    if (!this.isFitted) throw new Error("Scaler must be fitted before transforming.");
    return v.map((val, j) => (val - this.means[j]) / this.stds[j]);
  }

  toJSON() {
    return { means: this.means, stds: this.stds, isFitted: this.isFitted };
  }

  fromJSON(data: any) {
    this.means = data.means;
    this.stds = data.stds;
    this.isFitted = data.isFitted;
  }
}

// --- 3. Machine Learning Models ---

export interface Model {
  fit(X: number[][], y: number[]): void;
  predict(X: number[][]): number[];
  getFeatureImportance(featureNames: string[]): FeatureImportance[];
  toJSON(): any;
  fromJSON(json: any): void;
}

// 3.1 Linear Regression (Ordinary Least Squares)
export class LinearRegression implements Model {
  coefficients: number[] = []; // beta values, last one is intercept/bias
  isFitted = false;

  fit(X: number[][], y: number[]): void {
    const n = X.length;
    const m = X[0].length;

    // Create matrix Z = [X | 1] for intercept
    const Z = X.map(row => [...row, 1]);

    // OLS: beta = (Z^T * Z)^-1 * Z^T * y
    const Z_T = Matrix.transpose(Z);
    const Z_T_Z = Matrix.multiply(Z_T, Z);
    const Z_T_y = Matrix.multiplyMV(Z_T, y);

    try {
      const inv = Matrix.invert(Z_T_Z);
      this.coefficients = Matrix.multiplyMV(inv, Z_T_y);
      this.isFitted = true;
    } catch (e) {
      // Fallback: Gradient descent if matrix singular (unlikely with standardized socioeconomic indicators)
      console.warn("Singular matrix in OLS regression. Falling back to ridge/gradient descent.", e);
      this.fitGradientDescent(X, y);
    }
  }

  private fitGradientDescent(X: number[][], y: number[]): void {
    const n = X.length;
    const m = X[0].length;
    this.coefficients = new Array(m + 1).fill(0);
    const lr = 0.05;
    const epochs = 1000;

    for (let epoch = 0; epoch < epochs; epoch++) {
      const gradients = new Array(m + 1).fill(0);
      for (let i = 0; i < n; i++) {
        let pred = this.coefficients[m]; // intercept
        for (let j = 0; j < m; j++) {
          pred += X[i][j] * this.coefficients[j];
        }
        const error = pred - y[i];
        for (let j = 0; j < m; j++) {
          gradients[j] += error * X[i][j];
        }
        gradients[m] += error; // bias gradient
      }

      for (let j = 0; j <= m; j++) {
        this.coefficients[j] -= (lr * gradients[j]) / n;
      }
    }
    this.isFitted = true;
  }

  predict(X: number[][]): number[] {
    if (!this.isFitted) throw new Error("Model must be fitted first.");
    const m = X[0].length;
    return X.map(row => {
      let pred = this.coefficients[m]; // bias / intercept
      for (let j = 0; j < m; j++) {
        pred += row[j] * this.coefficients[j];
      }
      return pred;
    });
  }

  getFeatureImportance(featureNames: string[]): FeatureImportance[] {
    if (!this.isFitted) return [];
    // Standardized coefficients magnitude represents relative feature importance!
    const importances = featureNames.map((name, i) => ({
      feature: name,
      importance: Math.abs(this.coefficients[i])
    }));
    const total = importances.reduce((sum, item) => sum + item.importance, 0);
    return importances.map(item => ({
      feature: item.feature,
      importance: total > 0 ? item.importance / total : 0.25
    })).sort((a, b) => b.importance - a.importance);
  }

  toJSON() {
    return { coefficients: this.coefficients, isFitted: this.isFitted, type: "LinearRegression" };
  }

  fromJSON(json: any) {
    this.coefficients = json.coefficients;
    this.isFitted = json.isFitted;
  }
}

// 3.2 Decision Tree Regressor
interface TreeNode {
  featureIndex?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  value?: number; // average value of samples in node (only for leaves)
}

export class DecisionTreeRegressor implements Model {
  root: TreeNode | null = null;
  maxDepth: number;
  minSamplesSplit: number;

  constructor(maxDepth = 4, minSamplesSplit = 5) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
  }

  fit(X: number[][], y: number[]): void {
    const indices = Array.from({ length: X.length }, (_, i) => i);
    this.root = this.buildTree(X, y, indices, 0);
  }

  private buildTree(X: number[][], y: number[], indices: number[], depth: number): TreeNode {
    const numSamples = indices.length;

    // Base cases
    if (depth >= this.maxDepth || numSamples < this.minSamplesSplit) {
      return { value: this.calculateMean(y, indices) };
    }

    let bestVarReduction = -1;
    let bestFeature = -1;
    let bestThreshold = 0;
    let bestLeftIndices: number[] = [];
    let bestRightIndices: number[] = [];

    const parentVariance = this.calculateVariance(y, indices);
    const m = X[0].length;

    for (let f = 0; f < m; f++) {
      // Extract unique values of feature f for split candidates
      const values = indices.map(idx => X[idx][f]);
      const uniqueSorted = Array.from(new Set(values)).sort((a, b) => a - b);

      // Evaluate candidate splits (midpoints between consecutive sorted unique values)
      for (let i = 0; i < uniqueSorted.length - 1; i++) {
        const threshold = (uniqueSorted[i] + uniqueSorted[i + 1]) / 2;
        const left: number[] = [];
        const right: number[] = [];

        for (const idx of indices) {
          if (X[idx][f] <= threshold) {
            left.push(idx);
          } else {
            right.push(idx);
          }
        }

        if (left.length === 0 || right.length === 0) continue;

        // Calculate variance reduction
        const leftVar = this.calculateVariance(y, left);
        const rightVar = this.calculateVariance(y, right);
        const weightedChildVar = (left.length / numSamples) * leftVar + (right.length / numSamples) * rightVar;
        const varianceReduction = parentVariance - weightedChildVar;

        if (varianceReduction > bestVarReduction) {
          bestVarReduction = varianceReduction;
          bestFeature = f;
          bestThreshold = threshold;
          bestLeftIndices = left;
          bestRightIndices = right;
        }
      }
    }

    // If split does not decrease variance significantly
    if (bestVarReduction <= 1e-7 || bestFeature === -1) {
      return { value: this.calculateMean(y, indices) };
    }

    // Recursively build children
    const leftChild = this.buildTree(X, y, bestLeftIndices, depth + 1);
    const rightChild = this.buildTree(X, y, bestRightIndices, depth + 1);

    return {
      featureIndex: bestFeature,
      threshold: bestThreshold,
      left: leftChild,
      right: rightChild
    };
  }

  private calculateMean(y: number[], indices: number[]): number {
    if (indices.length === 0) return 0;
    let sum = 0;
    for (const idx of indices) sum += y[idx];
    return sum / indices.length;
  }

  private calculateVariance(y: number[], indices: number[]): number {
    const n = indices.length;
    if (n <= 1) return 0;
    const mean = this.calculateMean(y, indices);
    let sumSqDiff = 0;
    for (const idx of indices) {
      const diff = y[idx] - mean;
      sumSqDiff += diff * diff;
    }
    return sumSqDiff / n;
  }

  predict(X: number[][]): number[] {
    if (!this.root) throw new Error("Model must be fitted first.");
    return X.map(row => this.predictRow(this.root!, row));
  }

  private predictRow(node: TreeNode, row: number[]): number {
    if (node.value !== undefined) {
      return node.value;
    }
    if (row[node.featureIndex!] <= node.threshold!) {
      return this.predictRow(node.left!, row);
    } else {
      return this.predictRow(node.right!, row);
    }
  }

  getFeatureImportance(featureNames: string[]): FeatureImportance[] {
    const importances = new Array(featureNames.length).fill(0);
    if (!this.root) return [];
    
    // Calculate feature importances by traversing tree and accumulating variance reduction
    const traverse = (node: TreeNode) => {
      if (node.value !== undefined) return;
      // Approximate importance based on node occurrences and depth split
      importances[node.featureIndex!] += 1.0;
      traverse(node.left!);
      traverse(node.right!);
    };
    
    traverse(this.root);
    const total = importances.reduce((sum, val) => sum + val, 0);
    const result = featureNames.map((name, i) => ({
      feature: name,
      importance: total > 0 ? importances[i] / total : 0.25
    }));
    return result.sort((a, b) => b.importance - a.importance);
  }

  toJSON() {
    return { root: this.root, maxDepth: this.maxDepth, minSamplesSplit: this.minSamplesSplit, type: "DecisionTree" };
  }

  fromJSON(json: any) {
    this.root = json.root;
    this.maxDepth = json.maxDepth;
    this.minSamplesSplit = json.minSamplesSplit;
  }
}

// 3.3 Random Forest Regressor
export class RandomForestRegressor implements Model {
  trees: DecisionTreeRegressor[] = [];
  nEstimators: number;
  maxDepth: number;
  minSamplesSplit: number;

  constructor(nEstimators = 15, maxDepth = 4, minSamplesSplit = 5) {
    this.nEstimators = nEstimators;
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
  }

  fit(X: number[][], y: number[]): void {
    this.trees = [];
    const n = X.length;

    for (let i = 0; i < this.nEstimators; i++) {
      // Bootstrapping: draw samples with replacement
      const bootstrapX: number[][] = [];
      const bootstrapY: number[] = [];
      for (let j = 0; j < n; j++) {
        const randIdx = Math.floor(Math.random() * n);
        bootstrapX.push(X[randIdx]);
        bootstrapY.push(y[randIdx]);
      }

      const tree = new DecisionTreeRegressor(this.maxDepth, this.minSamplesSplit);
      tree.fit(bootstrapX, bootstrapY);
      this.trees.push(tree);
    }
  }

  predict(X: number[][]): number[] {
    if (this.trees.length === 0) throw new Error("Model must be fitted first.");
    const numTrees = this.trees.length;
    const allPredictions = this.trees.map(tree => tree.predict(X));
    const n = X.length;
    const finalPredictions = new Array(n).fill(0);

    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let t = 0; t < numTrees; t++) {
        sum += allPredictions[t][i];
      }
      finalPredictions[i] = sum / numTrees;
    }
    return finalPredictions;
  }

  getFeatureImportance(featureNames: string[]): FeatureImportance[] {
    if (this.trees.length === 0) return [];
    const importances = new Array(featureNames.length).fill(0);

    for (const tree of this.trees) {
      const treeImp = tree.getFeatureImportance(featureNames);
      for (const item of treeImp) {
        const idx = featureNames.indexOf(item.feature);
        if (idx !== -1) {
          importances[idx] += item.importance;
        }
      }
    }

    const total = importances.reduce((sum, val) => sum + val, 0);
    return featureNames.map((name, i) => ({
      feature: name,
      importance: total > 0 ? importances[i] / total : 0.25
    })).sort((a, b) => b.importance - a.importance);
  }

  toJSON() {
    return {
      trees: this.trees.map(tree => tree.toJSON()),
      nEstimators: this.nEstimators,
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      type: "RandomForest"
    };
  }

  fromJSON(json: any) {
    this.nEstimators = json.nEstimators;
    this.maxDepth = json.maxDepth;
    this.minSamplesSplit = json.minSamplesSplit;
    this.trees = json.trees.map((treeData: any) => {
      const tree = new DecisionTreeRegressor();
      tree.fromJSON(treeData);
      return tree;
    });
  }
}

// 3.4 Gradient Boosting Regressor
export class GradientBoostingRegressor implements Model {
  initialPrediction = 0;
  trees: DecisionTreeRegressor[] = [];
  nEstimators: number;
  learningRate: number;
  maxDepth: number;

  constructor(nEstimators = 15, learningRate = 0.1, maxDepth = 3) {
    this.nEstimators = nEstimators;
    this.learningRate = learningRate;
    this.maxDepth = maxDepth;
  }

  fit(X: number[][], y: number[]): void {
    this.trees = [];
    const n = X.length;

    // Calculate initial prediction (mean of y)
    let sumY = 0;
    for (const val of y) sumY += val;
    this.initialPrediction = sumY / n;

    // Current ensemble predictions
    const currentPreds = new Array(n).fill(this.initialPrediction);

    for (let i = 0; i < this.nEstimators; i++) {
      // Calculate residuals
      const residuals = y.map((val, idx) => val - currentPreds[idx]);

      // Fit tree to residuals
      const tree = new DecisionTreeRegressor(this.maxDepth, 4);
      tree.fit(X, residuals);
      this.trees.push(tree);

      // Update ensemble predictions
      const treePreds = tree.predict(X);
      for (let j = 0; j < n; j++) {
        currentPreds[j] += this.learningRate * treePreds[j];
      }
    }
  }

  predict(X: number[][]): number[] {
    if (this.trees.length === 0) throw new Error("Model must be fitted first.");
    const n = X.length;
    const preds = new Array(n).fill(this.initialPrediction);

    for (const tree of this.trees) {
      const treePreds = tree.predict(X);
      for (let i = 0; i < n; i++) {
        preds[i] += this.learningRate * treePreds[i];
      }
    }
    return preds;
  }

  getFeatureImportance(featureNames: string[]): FeatureImportance[] {
    if (this.trees.length === 0) return [];
    const importances = new Array(featureNames.length).fill(0);

    for (const tree of this.trees) {
      const treeImp = tree.getFeatureImportance(featureNames);
      for (const item of treeImp) {
        const idx = featureNames.indexOf(item.feature);
        if (idx !== -1) {
          importances[idx] += item.importance;
        }
      }
    }

    const total = importances.reduce((sum, val) => sum + val, 0);
    return featureNames.map((name, i) => ({
      feature: name,
      importance: total > 0 ? importances[i] / total : 0.25
    })).sort((a, b) => b.importance - a.importance);
  }

  toJSON() {
    return {
      initialPrediction: this.initialPrediction,
      trees: this.trees.map(tree => tree.toJSON()),
      nEstimators: this.nEstimators,
      learningRate: this.learningRate,
      maxDepth: this.maxDepth,
      type: "GradientBoosting"
    };
  }

  fromJSON(json: any) {
    this.initialPrediction = json.initialPrediction;
    this.nEstimators = json.nEstimators;
    this.learningRate = json.learningRate;
    this.maxDepth = json.maxDepth;
    this.trees = json.trees.map((treeData: any) => {
      const tree = new DecisionTreeRegressor();
      tree.fromJSON(treeData);
      return tree;
    });
  }
}

// --- 4. Evaluation and Metrics Engine ---

export class EvaluationMetrics {
  static r2Score(actual: number[], predicted: number[]): number {
    const n = actual.length;
    let sumActual = 0;
    for (const val of actual) sumActual += val;
    const meanActual = sumActual / n;

    let ssResidual = 0;
    let ssTotal = 0;

    for (let i = 0; i < n; i++) {
      const diffResidual = actual[i] - predicted[i];
      ssResidual += diffResidual * diffResidual;

      const diffTotal = actual[i] - meanActual;
      ssTotal += diffTotal * diffTotal;
    }

    if (ssTotal === 0) return 0;
    return 1 - ssResidual / ssTotal;
  }

  static mae(actual: number[], predicted: number[]): number {
    let sumAbsErr = 0;
    const n = actual.length;
    for (let i = 0; i < n; i++) {
      sumAbsErr += Math.abs(actual[i] - predicted[i]);
    }
    return sumAbsErr / n;
  }

  static mse(actual: number[], predicted: number[]): number {
    let sumSqErr = 0;
    const n = actual.length;
    for (let i = 0; i < n; i++) {
      const diff = actual[i] - predicted[i];
      sumSqErr += diff * diff;
    }
    return sumSqErr / n;
  }

  static rmse(actual: number[], predicted: number[]): number {
    return Math.sqrt(this.mse(actual, predicted));
  }

  static evaluate(actual: number[], predicted: number[]): ModelMetrics {
    return {
      r2: this.r2Score(actual, predicted),
      mae: this.mae(actual, predicted),
      mse: this.mse(actual, predicted),
      rmse: this.rmse(actual, predicted),
    };
  }
}

// --- 5. High-Level ML Pipeline Wrapper ---

export class HDIPipeline {
  scaler: StandardScaler = new StandardScaler();
  models: { [key: string]: Model } = {};
  bestModelName = "";
  featureNames = ["Life_Expectancy", "Expected_Schooling", "Mean_Schooling", "GNI_Per_Capita"];
  isTrained = false;

  constructor() {
    this.models = {
      "Linear Regression": new LinearRegression(),
      "Decision Tree Regressor": new DecisionTreeRegressor(4, 5),
      "Random Forest Regressor": new RandomForestRegressor(15, 4, 5),
      "Gradient Boosting Regressor": new GradientBoostingRegressor(15, 0.1, 3)
    };
  }

  train(X: number[][], y: number[]): { [key: string]: TrainingResult } {
    const numSamples = X.length;
    
    // Train-test split (75% Train, 25% Test)
    const splitIndex = Math.floor(numSamples * 0.75);
    
    // Let's shuffle first with a fixed seed for reproducible splits
    const indices = Array.from({ length: numSamples }, (_, i) => i);
    // Seeded shuffle helper
    let seed = 42;
    const randomSeeded = () => {
      const x = Math.sin(seed++) * 10000;
      return x - Math.floor(x);
    };
    
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(randomSeeded() * (i + 1));
      const temp = indices[i];
      indices[i] = indices[j];
      indices[j] = temp;
    }

    const trainIndices = indices.slice(0, splitIndex);
    const testIndices = indices.slice(splitIndex);

    const X_train = trainIndices.map(idx => X[idx]);
    const y_train = trainIndices.map(idx => y[idx]);
    const X_test = testIndices.map(idx => X[idx]);
    const y_test = testIndices.map(idx => y[idx]);

    // Fit StandardScaler on training data
    this.scaler.fit(X_train);
    const X_train_scaled = this.scaler.transform(X_train);
    const X_test_scaled = this.scaler.transform(X_test);

    const results: { [key: string]: TrainingResult } = {};
    let bestR2 = -Infinity;

    for (const [name, model] of Object.entries(this.models)) {
      // Fit model on scaled training data
      model.fit(X_train_scaled, y_train);
      
      // Predict on scaled test data
      const test_preds = model.predict(X_test_scaled);
      
      // Calculate evaluation metrics
      const metrics = EvaluationMetrics.evaluate(y_test, test_preds);
      const residuals = y_test.map((act, i) => act - test_preds[i]);

      results[name] = {
        modelName: name,
        metrics,
        predictions: test_preds,
        residuals
      };

      if (metrics.r2 > bestR2) {
        bestR2 = metrics.r2;
        this.bestModelName = name;
      }
    }

    this.isTrained = true;
    return results;
  }

  predict(v: number[]): { score: number; category: string; formattedScore: string } {
    if (!this.isTrained) throw new Error("Pipeline must be trained before predicting.");
    const bestModel = this.models[this.bestModelName];
    if (!bestModel) throw new Error("Best model not found.");

    // Scale input
    const scaledVector = this.scaler.transformVector(v);
    
    // Predict
    const pred = bestModel.predict([scaledVector])[0];
    
    // Clip prediction to valid HDI bounds [0.0, 1.0]
    const clampedPred = Math.max(0.0, Math.min(1.0, pred));
    
    // Classify predicted score
    let category = "Low";
    if (clampedPred >= 0.800) {
      category = "Very High";
    } else if (clampedPred >= 0.700) {
      category = "High";
    } else if (clampedPred >= 0.550) {
      category = "Medium";
    } else {
      category = "Low";
    }

    return {
      score: clampedPred,
      category,
      formattedScore: clampedPred.toFixed(3)
    };
  }

  getBestModelImportance(): FeatureImportance[] {
    const bestModel = this.models[this.bestModelName];
    if (!bestModel) return [];
    return bestModel.getFeatureImportance(this.featureNames);
  }

  toJSON() {
    const serializedModels: { [key: string]: any } = {};
    for (const [name, model] of Object.entries(this.models)) {
      serializedModels[name] = model.toJSON();
    }
    return {
      scaler: this.scaler.toJSON(),
      models: serializedModels,
      bestModelName: this.bestModelName,
      isTrained: this.isTrained,
    };
  }

  fromJSON(json: any) {
    this.scaler = new StandardScaler();
    this.scaler.fromJSON(json.scaler);
    this.bestModelName = json.bestModelName;
    this.isTrained = json.isTrained;

    for (const [name, modelData] of Object.entries(json.models)) {
      const type = (modelData as any).type;
      let model: Model;
      if (type === "LinearRegression") {
        model = new LinearRegression();
      } else if (type === "DecisionTree") {
        model = new DecisionTreeRegressor();
      } else if (type === "RandomForest") {
        model = new RandomForestRegressor();
      } else {
        model = new GradientBoostingRegressor();
      }
      model.fromJSON(modelData);
      this.models[name] = model;
    }
  }
}
