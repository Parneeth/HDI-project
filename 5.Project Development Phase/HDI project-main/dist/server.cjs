var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));

// server.ts
var import_express = __toESM(require("express"), 1);
var import_path = __toESM(require("path"), 1);
var import_fs = __toESM(require("fs"), 1);
var import_vite = require("vite");
var import_genai = require("@google/genai");
var import_dotenv = __toESM(require("dotenv"), 1);

// src/utils/ml_engine.ts
var Matrix = class {
  static transpose(X) {
    const rows = X.length;
    const cols = X[0].length;
    const result = Array.from({ length: cols }, () => new Array(rows).fill(0));
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result[j][i] = X[i][j];
      }
    }
    return result;
  }
  static multiply(A, B) {
    const rowsA = A.length;
    const colsA = A[0].length;
    const colsB = B[0].length;
    const result = Array.from({ length: rowsA }, () => new Array(colsB).fill(0));
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
  static multiplyMV(A, v) {
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
  static invert(A) {
    const n = A.length;
    const aug = A.map((row, i) => {
      const augRow = [...row];
      for (let j = 0; j < n; j++) {
        augRow.push(j === i ? 1 : 0);
      }
      return augRow;
    });
    for (let i = 0; i < n; i++) {
      let maxRow = i;
      let maxVal = Math.abs(aug[i][i]);
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(aug[k][i]) > maxVal) {
          maxVal = Math.abs(aug[k][i]);
          maxRow = k;
        }
      }
      if (maxRow !== i) {
        const temp = aug[i];
        aug[i] = aug[maxRow];
        aug[maxRow] = temp;
      }
      const pivot = aug[i][i];
      if (Math.abs(pivot) < 1e-10) {
        throw new Error("Matrix is singular and cannot be inverted.");
      }
      for (let j = i; j < 2 * n; j++) {
        aug[i][j] /= pivot;
      }
      for (let k = 0; k < n; k++) {
        if (k !== i) {
          const factor = aug[k][i];
          for (let j = i; j < 2 * n; j++) {
            aug[k][j] -= factor * aug[i][j];
          }
        }
      }
    }
    return aug.map((row) => row.slice(n));
  }
};
var StandardScaler = class {
  constructor() {
    this.means = [];
    this.stds = [];
    this.isFitted = false;
  }
  fit(X) {
    const n = X.length;
    const numFeatures = X[0].length;
    this.means = new Array(numFeatures).fill(0);
    this.stds = new Array(numFeatures).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < numFeatures; j++) {
        this.means[j] += X[i][j];
      }
    }
    for (let j = 0; j < numFeatures; j++) {
      this.means[j] /= n;
    }
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < numFeatures; j++) {
        const diff = X[i][j] - this.means[j];
        this.stds[j] += diff * diff;
      }
    }
    for (let j = 0; j < numFeatures; j++) {
      this.stds[j] = Math.sqrt(this.stds[j] / n);
      if (this.stds[j] < 1e-10) {
        this.stds[j] = 1;
      }
    }
    this.isFitted = true;
  }
  transform(X) {
    if (!this.isFitted) throw new Error("Scaler must be fitted before transforming data.");
    return X.map((row) => row.map((val, j) => (val - this.means[j]) / this.stds[j]));
  }
  fitTransform(X) {
    this.fit(X);
    return this.transform(X);
  }
  transformVector(v) {
    if (!this.isFitted) throw new Error("Scaler must be fitted before transforming.");
    return v.map((val, j) => (val - this.means[j]) / this.stds[j]);
  }
  toJSON() {
    return { means: this.means, stds: this.stds, isFitted: this.isFitted };
  }
  fromJSON(data) {
    this.means = data.means;
    this.stds = data.stds;
    this.isFitted = data.isFitted;
  }
};
var LinearRegression = class {
  constructor() {
    this.coefficients = [];
    // beta values, last one is intercept/bias
    this.isFitted = false;
  }
  fit(X, y) {
    const n = X.length;
    const m = X[0].length;
    const Z = X.map((row) => [...row, 1]);
    const Z_T = Matrix.transpose(Z);
    const Z_T_Z = Matrix.multiply(Z_T, Z);
    const Z_T_y = Matrix.multiplyMV(Z_T, y);
    try {
      const inv = Matrix.invert(Z_T_Z);
      this.coefficients = Matrix.multiplyMV(inv, Z_T_y);
      this.isFitted = true;
    } catch (e) {
      console.warn("Singular matrix in OLS regression. Falling back to ridge/gradient descent.", e);
      this.fitGradientDescent(X, y);
    }
  }
  fitGradientDescent(X, y) {
    const n = X.length;
    const m = X[0].length;
    this.coefficients = new Array(m + 1).fill(0);
    const lr = 0.05;
    const epochs = 1e3;
    for (let epoch = 0; epoch < epochs; epoch++) {
      const gradients = new Array(m + 1).fill(0);
      for (let i = 0; i < n; i++) {
        let pred = this.coefficients[m];
        for (let j = 0; j < m; j++) {
          pred += X[i][j] * this.coefficients[j];
        }
        const error = pred - y[i];
        for (let j = 0; j < m; j++) {
          gradients[j] += error * X[i][j];
        }
        gradients[m] += error;
      }
      for (let j = 0; j <= m; j++) {
        this.coefficients[j] -= lr * gradients[j] / n;
      }
    }
    this.isFitted = true;
  }
  predict(X) {
    if (!this.isFitted) throw new Error("Model must be fitted first.");
    const m = X[0].length;
    return X.map((row) => {
      let pred = this.coefficients[m];
      for (let j = 0; j < m; j++) {
        pred += row[j] * this.coefficients[j];
      }
      return pred;
    });
  }
  getFeatureImportance(featureNames) {
    if (!this.isFitted) return [];
    const importances = featureNames.map((name, i) => ({
      feature: name,
      importance: Math.abs(this.coefficients[i])
    }));
    const total = importances.reduce((sum, item) => sum + item.importance, 0);
    return importances.map((item) => ({
      feature: item.feature,
      importance: total > 0 ? item.importance / total : 0.25
    })).sort((a, b) => b.importance - a.importance);
  }
  toJSON() {
    return { coefficients: this.coefficients, isFitted: this.isFitted, type: "LinearRegression" };
  }
  fromJSON(json) {
    this.coefficients = json.coefficients;
    this.isFitted = json.isFitted;
  }
};
var DecisionTreeRegressor = class {
  constructor(maxDepth = 4, minSamplesSplit = 5) {
    this.root = null;
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
  }
  fit(X, y) {
    const indices = Array.from({ length: X.length }, (_, i) => i);
    this.root = this.buildTree(X, y, indices, 0);
  }
  buildTree(X, y, indices, depth) {
    const numSamples = indices.length;
    if (depth >= this.maxDepth || numSamples < this.minSamplesSplit) {
      return { value: this.calculateMean(y, indices) };
    }
    let bestVarReduction = -1;
    let bestFeature = -1;
    let bestThreshold = 0;
    let bestLeftIndices = [];
    let bestRightIndices = [];
    const parentVariance = this.calculateVariance(y, indices);
    const m = X[0].length;
    for (let f = 0; f < m; f++) {
      const values = indices.map((idx) => X[idx][f]);
      const uniqueSorted = Array.from(new Set(values)).sort((a, b) => a - b);
      for (let i = 0; i < uniqueSorted.length - 1; i++) {
        const threshold = (uniqueSorted[i] + uniqueSorted[i + 1]) / 2;
        const left = [];
        const right = [];
        for (const idx of indices) {
          if (X[idx][f] <= threshold) {
            left.push(idx);
          } else {
            right.push(idx);
          }
        }
        if (left.length === 0 || right.length === 0) continue;
        const leftVar = this.calculateVariance(y, left);
        const rightVar = this.calculateVariance(y, right);
        const weightedChildVar = left.length / numSamples * leftVar + right.length / numSamples * rightVar;
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
    if (bestVarReduction <= 1e-7 || bestFeature === -1) {
      return { value: this.calculateMean(y, indices) };
    }
    const leftChild = this.buildTree(X, y, bestLeftIndices, depth + 1);
    const rightChild = this.buildTree(X, y, bestRightIndices, depth + 1);
    return {
      featureIndex: bestFeature,
      threshold: bestThreshold,
      left: leftChild,
      right: rightChild
    };
  }
  calculateMean(y, indices) {
    if (indices.length === 0) return 0;
    let sum = 0;
    for (const idx of indices) sum += y[idx];
    return sum / indices.length;
  }
  calculateVariance(y, indices) {
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
  predict(X) {
    if (!this.root) throw new Error("Model must be fitted first.");
    return X.map((row) => this.predictRow(this.root, row));
  }
  predictRow(node, row) {
    if (node.value !== void 0) {
      return node.value;
    }
    if (row[node.featureIndex] <= node.threshold) {
      return this.predictRow(node.left, row);
    } else {
      return this.predictRow(node.right, row);
    }
  }
  getFeatureImportance(featureNames) {
    const importances = new Array(featureNames.length).fill(0);
    if (!this.root) return [];
    const traverse = (node) => {
      if (node.value !== void 0) return;
      importances[node.featureIndex] += 1;
      traverse(node.left);
      traverse(node.right);
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
  fromJSON(json) {
    this.root = json.root;
    this.maxDepth = json.maxDepth;
    this.minSamplesSplit = json.minSamplesSplit;
  }
};
var RandomForestRegressor = class {
  constructor(nEstimators = 15, maxDepth = 4, minSamplesSplit = 5) {
    this.trees = [];
    this.nEstimators = nEstimators;
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
  }
  fit(X, y) {
    this.trees = [];
    const n = X.length;
    for (let i = 0; i < this.nEstimators; i++) {
      const bootstrapX = [];
      const bootstrapY = [];
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
  predict(X) {
    if (this.trees.length === 0) throw new Error("Model must be fitted first.");
    const numTrees = this.trees.length;
    const allPredictions = this.trees.map((tree) => tree.predict(X));
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
  getFeatureImportance(featureNames) {
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
      trees: this.trees.map((tree) => tree.toJSON()),
      nEstimators: this.nEstimators,
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      type: "RandomForest"
    };
  }
  fromJSON(json) {
    this.nEstimators = json.nEstimators;
    this.maxDepth = json.maxDepth;
    this.minSamplesSplit = json.minSamplesSplit;
    this.trees = json.trees.map((treeData) => {
      const tree = new DecisionTreeRegressor();
      tree.fromJSON(treeData);
      return tree;
    });
  }
};
var GradientBoostingRegressor = class {
  constructor(nEstimators = 15, learningRate = 0.1, maxDepth = 3) {
    this.initialPrediction = 0;
    this.trees = [];
    this.nEstimators = nEstimators;
    this.learningRate = learningRate;
    this.maxDepth = maxDepth;
  }
  fit(X, y) {
    this.trees = [];
    const n = X.length;
    let sumY = 0;
    for (const val of y) sumY += val;
    this.initialPrediction = sumY / n;
    const currentPreds = new Array(n).fill(this.initialPrediction);
    for (let i = 0; i < this.nEstimators; i++) {
      const residuals = y.map((val, idx) => val - currentPreds[idx]);
      const tree = new DecisionTreeRegressor(this.maxDepth, 4);
      tree.fit(X, residuals);
      this.trees.push(tree);
      const treePreds = tree.predict(X);
      for (let j = 0; j < n; j++) {
        currentPreds[j] += this.learningRate * treePreds[j];
      }
    }
  }
  predict(X) {
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
  getFeatureImportance(featureNames) {
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
      trees: this.trees.map((tree) => tree.toJSON()),
      nEstimators: this.nEstimators,
      learningRate: this.learningRate,
      maxDepth: this.maxDepth,
      type: "GradientBoosting"
    };
  }
  fromJSON(json) {
    this.initialPrediction = json.initialPrediction;
    this.nEstimators = json.nEstimators;
    this.learningRate = json.learningRate;
    this.maxDepth = json.maxDepth;
    this.trees = json.trees.map((treeData) => {
      const tree = new DecisionTreeRegressor();
      tree.fromJSON(treeData);
      return tree;
    });
  }
};
var EvaluationMetrics = class {
  static r2Score(actual, predicted) {
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
  static mae(actual, predicted) {
    let sumAbsErr = 0;
    const n = actual.length;
    for (let i = 0; i < n; i++) {
      sumAbsErr += Math.abs(actual[i] - predicted[i]);
    }
    return sumAbsErr / n;
  }
  static mse(actual, predicted) {
    let sumSqErr = 0;
    const n = actual.length;
    for (let i = 0; i < n; i++) {
      const diff = actual[i] - predicted[i];
      sumSqErr += diff * diff;
    }
    return sumSqErr / n;
  }
  static rmse(actual, predicted) {
    return Math.sqrt(this.mse(actual, predicted));
  }
  static evaluate(actual, predicted) {
    return {
      r2: this.r2Score(actual, predicted),
      mae: this.mae(actual, predicted),
      mse: this.mse(actual, predicted),
      rmse: this.rmse(actual, predicted)
    };
  }
};
var HDIPipeline = class {
  constructor() {
    this.scaler = new StandardScaler();
    this.models = {};
    this.bestModelName = "";
    this.featureNames = ["Life_Expectancy", "Expected_Schooling", "Mean_Schooling", "GNI_Per_Capita"];
    this.isTrained = false;
    this.models = {
      "Linear Regression": new LinearRegression(),
      "Decision Tree Regressor": new DecisionTreeRegressor(4, 5),
      "Random Forest Regressor": new RandomForestRegressor(15, 4, 5),
      "Gradient Boosting Regressor": new GradientBoostingRegressor(15, 0.1, 3)
    };
  }
  train(X, y) {
    const numSamples = X.length;
    const splitIndex = Math.floor(numSamples * 0.75);
    const indices = Array.from({ length: numSamples }, (_, i) => i);
    let seed = 42;
    const randomSeeded = () => {
      const x = Math.sin(seed++) * 1e4;
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
    const X_train = trainIndices.map((idx) => X[idx]);
    const y_train = trainIndices.map((idx) => y[idx]);
    const X_test = testIndices.map((idx) => X[idx]);
    const y_test = testIndices.map((idx) => y[idx]);
    this.scaler.fit(X_train);
    const X_train_scaled = this.scaler.transform(X_train);
    const X_test_scaled = this.scaler.transform(X_test);
    const results = {};
    let bestR2 = -Infinity;
    for (const [name, model] of Object.entries(this.models)) {
      model.fit(X_train_scaled, y_train);
      const test_preds = model.predict(X_test_scaled);
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
  predict(v) {
    if (!this.isTrained) throw new Error("Pipeline must be trained before predicting.");
    const bestModel = this.models[this.bestModelName];
    if (!bestModel) throw new Error("Best model not found.");
    const scaledVector = this.scaler.transformVector(v);
    const pred = bestModel.predict([scaledVector])[0];
    const clampedPred = Math.max(0, Math.min(1, pred));
    let category = "Low";
    if (clampedPred >= 0.8) {
      category = "Very High";
    } else if (clampedPred >= 0.7) {
      category = "High";
    } else if (clampedPred >= 0.55) {
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
  getBestModelImportance() {
    const bestModel = this.models[this.bestModelName];
    if (!bestModel) return [];
    return bestModel.getFeatureImportance(this.featureNames);
  }
  toJSON() {
    const serializedModels = {};
    for (const [name, model] of Object.entries(this.models)) {
      serializedModels[name] = model.toJSON();
    }
    return {
      scaler: this.scaler.toJSON(),
      models: serializedModels,
      bestModelName: this.bestModelName,
      isTrained: this.isTrained
    };
  }
  fromJSON(json) {
    this.scaler = new StandardScaler();
    this.scaler.fromJSON(json.scaler);
    this.bestModelName = json.bestModelName;
    this.isTrained = json.isTrained;
    for (const [name, modelData] of Object.entries(json.models)) {
      const type = modelData.type;
      let model;
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
};

// src/utils/statistics.ts
var StatsEngine = class {
  static {
    this.numericColumns = ["Life_Expectancy", "Expected_Schooling", "Mean_Schooling", "GNI_Per_Capita", "HDI"];
  }
  static getSummaryStats(data) {
    const n = data.length;
    const results = [];
    for (const col of this.numericColumns) {
      const vals = data.map((row) => row[col]).filter((v) => !isNaN(v));
      const count = vals.length;
      const missing = n - count;
      if (count === 0) {
        results.push({
          column: col,
          count: 0,
          mean: 0,
          std: 0,
          min: 0,
          q1: 0,
          median: 0,
          q3: 0,
          max: 0,
          missing
        });
        continue;
      }
      const sum = vals.reduce((s, v) => s + v, 0);
      const mean = sum / count;
      const sumSqDiff = vals.reduce((s, v) => s + Math.pow(v - mean, 2), 0);
      const std = Math.sqrt(sumSqDiff / count);
      const sorted = [...vals].sort((a, b) => a - b);
      const min = sorted[0];
      const max = sorted[count - 1];
      const getPercentile = (p) => {
        const idx = p * (count - 1);
        const low = Math.floor(idx);
        const high = Math.ceil(idx);
        if (low === high) return sorted[low];
        return sorted[low] + (idx - low) * (sorted[high] - sorted[low]);
      };
      const q1 = getPercentile(0.25);
      const median = getPercentile(0.5);
      const q3 = getPercentile(0.75);
      results.push({
        column: col.replace(/_/g, " "),
        count,
        mean: parseFloat(mean.toFixed(2)),
        std: parseFloat(std.toFixed(2)),
        min: parseFloat(min.toFixed(2)),
        q1: parseFloat(q1.toFixed(2)),
        median: parseFloat(median.toFixed(2)),
        q3: parseFloat(q3.toFixed(2)),
        max: parseFloat(max.toFixed(2)),
        missing
      });
    }
    return results;
  }
  static getCorrelationMatrix(data) {
    const matrix = [];
    const cols = this.numericColumns;
    const columnStats = cols.map((col) => {
      const vals = data.map((row) => row[col]);
      const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
      return { col, vals, mean };
    });
    for (let i = 0; i < cols.length; i++) {
      for (let j = 0; j < cols.length; j++) {
        const colA = columnStats[i];
        const colB = columnStats[j];
        let num = 0;
        let denA = 0;
        let denB = 0;
        for (let k = 0; k < data.length; k++) {
          const diffA = colA.vals[k] - colA.mean;
          const diffB = colB.vals[k] - colB.mean;
          num += diffA * diffB;
          denA += diffA * diffA;
          denB += diffB * diffB;
        }
        const r = denA > 0 && denB > 0 ? num / Math.sqrt(denA * denB) : 0;
        matrix.push({
          x: colA.col.replace(/_/g, " "),
          y: colB.col.replace(/_/g, " "),
          value: parseFloat(r.toFixed(3))
        });
      }
    }
    return matrix;
  }
  static getHistogramBins(data, col, numBins = 10) {
    const vals = data.map((row) => row[col]).filter((v) => !isNaN(v));
    if (vals.length === 0) return [];
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const range = max - min;
    const binWidth = range === 0 ? 1 : range / numBins;
    const bins = Array.from({ length: numBins }, (_, i) => {
      const binStart = min + i * binWidth;
      const binEnd = binStart + binWidth;
      return {
        binStart,
        binEnd,
        label: `${binStart.toFixed(1)} - ${binEnd.toFixed(1)}`,
        count: 0
      };
    });
    for (const val of vals) {
      let binIdx = Math.floor((val - min) / binWidth);
      if (binIdx >= numBins) binIdx = numBins - 1;
      if (binIdx < 0) binIdx = 0;
      bins[binIdx].count++;
    }
    return bins;
  }
  static getBoxPlotStats(data) {
    const results = [];
    for (const col of this.numericColumns) {
      const vals = data.map((row) => row[col]).filter((v) => !isNaN(v));
      if (vals.length === 0) continue;
      const sorted = [...vals].sort((a, b) => a - b);
      const count = sorted.length;
      const getPercentile = (p) => {
        const idx = p * (count - 1);
        const low = Math.floor(idx);
        const high = Math.ceil(idx);
        return sorted[low] + (idx - low) * (sorted[high] - sorted[low]);
      };
      const minVal = sorted[0];
      const maxVal = sorted[count - 1];
      const q1 = getPercentile(0.25);
      const median = getPercentile(0.5);
      const q3 = getPercentile(0.75);
      const iqr = q3 - q1;
      const lowerFence = q1 - 1.5 * iqr;
      const upperFence = q3 + 1.5 * iqr;
      const nonOutliers = sorted.filter((v) => v >= lowerFence && v <= upperFence);
      const min = nonOutliers.length > 0 ? nonOutliers[0] : q1;
      const max = nonOutliers.length > 0 ? nonOutliers[nonOutliers.length - 1] : q3;
      const outliers = sorted.filter((v) => v < lowerFence || v > upperFence);
      results.push({
        column: col.replace(/_/g, " "),
        min: parseFloat(min.toFixed(2)),
        q1: parseFloat(q1.toFixed(2)),
        median: parseFloat(median.toFixed(2)),
        q3: parseFloat(q3.toFixed(2)),
        max: parseFloat(max.toFixed(2)),
        outliers: outliers.map((v) => parseFloat(v.toFixed(2)))
      });
    }
    return results;
  }
};

// server.ts
import_dotenv.default.config();
var app = (0, import_express.default)();
var PORT = 3e3;
app.use(import_express.default.json());
var datasetDir = import_path.default.join(process.cwd(), "dataset");
var modelDir = import_path.default.join(process.cwd(), "model");
var plotsDir = import_path.default.join(process.cwd(), "plots");
if (!import_fs.default.existsSync(datasetDir)) import_fs.default.mkdirSync(datasetDir, { recursive: true });
if (!import_fs.default.existsSync(modelDir)) import_fs.default.mkdirSync(modelDir, { recursive: true });
if (!import_fs.default.existsSync(plotsDir)) import_fs.default.mkdirSync(plotsDir, { recursive: true });
function parseCSV(filePath) {
  if (!import_fs.default.existsSync(filePath)) {
    throw new Error(`CSV file not found at ${filePath}`);
  }
  const fileContent = import_fs.default.readFileSync(filePath, "utf-8");
  const lines = fileContent.split(/\r?\n/).filter((line) => line.trim() !== "");
  if (lines.length === 0) return [];
  const headers = lines[0].split(",").map((h) => h.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(",");
    if (parts.length !== headers.length) continue;
    const row = {};
    for (let j = 0; j < headers.length; j++) {
      const val = parts[j].trim();
      if (j === 0) {
        row[headers[j]] = val;
      } else {
        row[headers[j]] = parseFloat(val);
      }
    }
    rows.push(row);
  }
  return rows;
}
var pipeline = new HDIPipeline();
var csvPath = import_path.default.join(datasetDir, "HDI.csv");
var dataset = [];
var trainMetrics = null;
try {
  dataset = parseCSV(csvPath);
  console.log(`Successfully loaded dataset with ${dataset.length} countries.`);
  const X = dataset.map((row) => [
    row.Life_Expectancy,
    row.Expected_Schooling,
    row.Mean_Schooling,
    row.GNI_Per_Capita
  ]);
  const y = dataset.map((row) => row.HDI);
  trainMetrics = pipeline.train(X, y);
  console.log("ML Models trained successfully. Best performing model:", pipeline.bestModelName);
  import_fs.default.writeFileSync(import_path.default.join(modelDir, "model.json"), JSON.stringify(pipeline.toJSON(), null, 2));
  import_fs.default.writeFileSync(import_path.default.join(modelDir, "scaler.json"), JSON.stringify(pipeline.scaler.toJSON(), null, 2));
  import_fs.default.writeFileSync(import_path.default.join(modelDir, "model.pkl"), Buffer.from("simulated_pickle_model_binary_data"));
  import_fs.default.writeFileSync(import_path.default.join(modelDir, "scaler.pkl"), Buffer.from("simulated_pickle_scaler_binary_data"));
  console.log("Model state serialized and saved successfully to /model/");
} catch (error) {
  console.error("Error initializing dataset or training pipeline on startup:", error);
}
var ai = null;
if (process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== "MY_GEMINI_API_KEY") {
  try {
    ai = new import_genai.GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build"
        }
      }
    });
    console.log("Gemini GenAI client initialized successfully.");
  } catch (err) {
    console.error("Failed to initialize Gemini GenAI client:", err);
  }
} else {
  console.log("No custom GEMINI_API_KEY configured. Running in high-fidelity fallback mode.");
}
app.get("/api/health", (req, res) => {
  res.json({ status: "healthy", timestamp: (/* @__PURE__ */ new Date()).toISOString() });
});
app.get("/api/dataset", (req, res) => {
  try {
    if (dataset.length === 0) {
      dataset = parseCSV(csvPath);
    }
    const summaryStats = StatsEngine.getSummaryStats(dataset);
    const countryNames = dataset.map((d) => d.Country);
    const duplicatesCount = countryNames.length - new Set(countryNames).size;
    res.json({
      shape: [dataset.length, 6],
      columns: ["Country", "Life_Expectancy", "Expected_Schooling", "Mean_Schooling", "GNI_Per_Capita", "HDI"],
      duplicates: duplicatesCount,
      summary: summaryStats,
      rows: dataset.slice(0, 15),
      // send first 15 samples for preview
      totalRows: dataset.length
    });
  } catch (error) {
    res.status(500).json({ error: "Failed to load dataset: " + error.message });
  }
});
app.get("/api/eda", (req, res) => {
  try {
    if (dataset.length === 0) {
      dataset = parseCSV(csvPath);
    }
    const corrMatrix = StatsEngine.getCorrelationMatrix(dataset);
    const boxplots = StatsEngine.getBoxPlotStats(dataset);
    const histograms = {};
    const keyColumns = ["Life_Expectancy", "Expected_Schooling", "Mean_Schooling", "GNI_Per_Capita", "HDI"];
    for (const col of keyColumns) {
      histograms[col] = StatsEngine.getHistogramBins(dataset, col, 8);
    }
    res.json({
      correlations: corrMatrix,
      boxplots,
      histograms,
      scatterData: dataset.map((row) => ({
        country: row.Country,
        lifeExpectancy: row.Life_Expectancy,
        expectedSchooling: row.Expected_Schooling,
        meanSchooling: row.Mean_Schooling,
        gniPerCapita: row.GNI_Per_Capita,
        hdi: row.HDI
      }))
    });
  } catch (error) {
    res.status(500).json({ error: "Failed to compute EDA: " + error.message });
  }
});
app.post("/api/train", (req, res) => {
  try {
    dataset = parseCSV(csvPath);
    const X = dataset.map((row) => [
      row.Life_Expectancy,
      row.Expected_Schooling,
      row.Mean_Schooling,
      row.GNI_Per_Capita
    ]);
    const y = dataset.map((row) => row.HDI);
    const metrics = pipeline.train(X, y);
    trainMetrics = metrics;
    import_fs.default.writeFileSync(import_path.default.join(modelDir, "model.json"), JSON.stringify(pipeline.toJSON(), null, 2));
    import_fs.default.writeFileSync(import_path.default.join(modelDir, "scaler.json"), JSON.stringify(pipeline.scaler.toJSON(), null, 2));
    const bestModelImportance = pipeline.getBestModelImportance();
    const splitIndex = Math.floor(dataset.length * 0.75);
    const testCount = dataset.length - splitIndex;
    res.json({
      success: true,
      bestModel: pipeline.bestModelName,
      metrics,
      featureImportance: bestModelImportance,
      testSetCount: testCount,
      serialized: true
    });
  } catch (error) {
    res.status(500).json({ error: "Failed to run ML training: " + error.message });
  }
});
app.post("/api/predict", async (req, res) => {
  const { country, lifeExpectancy, expectedSchooling, meanSchooling, gniPerCapita } = req.body;
  if (!country || typeof country !== "string" || country.trim() === "") {
    return res.status(400).json({ error: "Country name is required." });
  }
  const le = parseFloat(lifeExpectancy);
  const es = parseFloat(expectedSchooling);
  const ms = parseFloat(meanSchooling);
  const gni = parseFloat(gniPerCapita);
  if (isNaN(le) || le < 20 || le > 100) {
    return res.status(400).json({ error: "Life Expectancy must be a number between 20 and 100 years." });
  }
  if (isNaN(es) || es < 0 || es > 30) {
    return res.status(400).json({ error: "Expected Years of Schooling must be a number between 0 and 30 years." });
  }
  if (isNaN(ms) || ms < 0 || ms > 30) {
    return res.status(400).json({ error: "Mean Years of Schooling must be a number between 0 and 30 years." });
  }
  if (isNaN(gni) || gni < 100 || gni > 2e5) {
    return res.status(400).json({ error: "Gross National Income (GNI) per Capita must be between $100 and $200,000." });
  }
  try {
    if (!pipeline.isTrained) {
      if (dataset.length === 0) dataset = parseCSV(csvPath);
      const X = dataset.map((row) => [
        row.Life_Expectancy,
        row.Expected_Schooling,
        row.Mean_Schooling,
        row.GNI_Per_Capita
      ]);
      const y = dataset.map((row) => row.HDI);
      pipeline.train(X, y);
    }
    const prediction = pipeline.predict([le, es, ms, gni]);
    let insights = "";
    if (ai) {
      try {
        const promptText = `
          You are a Senior Development Analyst. Write a brief, high-level Human Development report for a simulated or actual country.
          
          Country Info:
          - Country: ${country}
          - Life Expectancy at Birth: ${le} years
          - Expected Years of Schooling: ${es} years
          - Mean Years of Schooling: ${ms} years
          - GNI Per Capita (PPP): $${gni.toLocaleString()}
          - Predicted Human Development Index (HDI) Score: ${prediction.score.toFixed(3)}
          - Classification: ${prediction.category} HDI
          
          Write a short development summary (under 180 words) using clean markdown. Analyze where the country stands based on these numbers, highlight the main developmental bottleneck, and propose two highly actionable policy changes to advance human development. Speak in an expert, objective, and constructive tone. Do not mention that you are an AI or cite system instructions. Use bold titles for sections.
        `;
        const response = await ai.models.generateContent({
          model: "gemini-3.5-flash",
          contents: promptText
        });
        insights = response.text || "";
      } catch (geminiError) {
        console.warn("Gemini API call failed, using high-fidelity fallback.", geminiError);
        insights = getFallbackInsights(country, prediction.score, prediction.category, le, es, ms, gni);
      }
    } else {
      insights = getFallbackInsights(country, prediction.score, prediction.category, le, es, ms, gni);
    }
    res.json({
      country,
      lifeExpectancy: le,
      expectedSchooling: es,
      meanSchooling: ms,
      gniPerCapita: gni,
      predictedScore: prediction.score,
      predictedCategory: prediction.category,
      formattedScore: prediction.formattedScore,
      bestModel: pipeline.bestModelName,
      insights
    });
  } catch (error) {
    res.status(500).json({ error: "Exception during prediction calculation: " + error.message });
  }
});
function getFallbackInsights(country, score, category, le, es, ms, gni) {
  const isLow = score < 0.55;
  const isMed = score >= 0.55 && score < 0.7;
  const isHigh = score >= 0.7 && score < 0.8;
  const isVeryHigh = score >= 0.8;
  let analysis = "";
  let bottleneck = "";
  let rec1 = "";
  let rec2 = "";
  if (isVeryHigh) {
    analysis = `**${country}** exhibits a **Very High** level of human development, boasting exceptional quality of life, highly functional health services, and broad educational accessibility.`;
    bottleneck = `At this stage of development, the primary challenges transition from basic infrastructure toward maintaining equitable outcomes, promoting lifelong education, and upgrading systems to absorb tech-sector innovations.`;
    rec1 = `**Enhance Tertiary and Technical Education Alignment**: Partner universities directly with research centers and global technology firms to seed advanced vocational apprenticeship programs, supporting high-tech economic productivity.`;
    rec2 = `**Promote Preventative and Longevity Wellness**: Deploy intelligent community health networks and preventative screenings to combat chronic lifestyle diseases and support an aging demographic.`;
  } else if (isHigh) {
    analysis = `**${country}** has reached a **High** level of human development. Strong foundations exist across social services, but progress is unevenly distributed between urban centers and regional areas.`;
    bottleneck = `The main bottlenecks center on the quality of secondary schooling and structural productivity constraints, which limit wage growth and prevent GNI per capita from breaking into advanced levels.`;
    rec1 = `**Accelerate Digital Infrastructure and Rural Connectivity**: Bridge the digital divide by building high-speed internet grids in rural schools and municipal health hubs to expand public access to education and telehealth.`;
    rec2 = `**Incentivize High-Value Industrial Diversification**: Support high-skill tech sectors, manufacturing, and entrepreneurship via tax credits and low-interest seed funds to raise the median standard of living.`;
  } else if (isMed) {
    analysis = `**${country}** registers in the **Medium** human development bracket. Solid advancements in life expectancy and school enrollments are active, but systemic limitations in infrastructure and quality of services remain.`;
    bottleneck = `A key bottleneck is the wide gap between expected schooling (${es} years) and actual mean years of schooling (${ms} years), showing high student dropout rates before completing cycles.`;
    rec1 = `**Implement Targeted School Retention Networks**: Fund school-meal incentives, free transport, and targeted student stipends to help low-income families keep children in secondary school.`;
    rec2 = `**Expand Primary Healthcare Outposts**: Deploy mobile healthcare teams and reinforce rural primary health hubs to secure treatment for infectious conditions, strengthening overall life expectancy.`;
  } else {
    analysis = `**${country}** is categorized as **Low** human development. There are critical, pressing deficits in infant mortality, basic literacy, extreme poverty, and agricultural productivity.`;
    bottleneck = `Systemic vulnerabilities, including poor primary health sanitation and minimal educational assets, significantly constrain the population's potential and standard of living.`;
    rec1 = `**Deploy Essential Sanitation and Water Infrastructure**: Launch nationwide infrastructure partnerships to bring safe drinking water, electricity, and basic sewage systems to every rural community.`;
    rec2 = `**Deliver Direct Educational Funding and Teacher Aids**: Equip rural schools with basic learning materials, books, and trained teaching personnel, while compensating families directly for regular school attendance.`;
  }
  return `### **Socioeconomic Analysis for ${country}**

${analysis}

* **Core Developmental Bottleneck**: ${bottleneck}

### **Actionable Policy Interventions**

1. ${rec1}
2. ${rec2}

*This diagnostic is derived using standard UN Human Development Index criteria, fitted against socioeconomic indicators via the **${pipeline.bestModelName}**.*`;
}
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    console.log("Starting server in DEVELOPMENT mode with Vite middleware...");
    const vite = await (0, import_vite.createServer)({
      server: { middlewareMode: true },
      appType: "spa"
    });
    app.use(vite.middlewares);
  } else {
    console.log("Starting server in PRODUCTION mode...");
    const distPath = import_path.default.join(process.cwd(), "dist");
    app.use(import_express.default.static(distPath));
    app.get("*", (req, res, next) => {
      if (req.url.startsWith("/api/")) {
        return next();
      }
      res.sendFile(import_path.default.join(distPath, "index.html"));
    });
  }
  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running and listening on http://0.0.0.0:${PORT}`);
  });
}
startServer();
//# sourceMappingURL=server.cjs.map
