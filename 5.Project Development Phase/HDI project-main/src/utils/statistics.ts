/**
 * Statistics and EDA calculations helper
 * Computes descriptive statistics, correlations, histograms, 
 * boxplots, and scatterplots dynamically from dataset.
 */

export interface HDIRow {
  Country: string;
  Life_Expectancy: number;
  Expected_Schooling: number;
  Mean_Schooling: number;
  GNI_Per_Capita: number;
  HDI: number;
}

export interface SummaryStats {
  column: string;
  count: number;
  mean: number;
  std: number;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  missing: number;
}

export interface CorrelationCell {
  x: string;
  y: string;
  value: number;
}

export interface HistogramBin {
  binStart: number;
  binEnd: number;
  label: string;
  count: number;
}

export interface BoxPlotStats {
  column: string;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  outliers: number[];
}

export class StatsEngine {
  static numericColumns = ["Life_Expectancy", "Expected_Schooling", "Mean_Schooling", "GNI_Per_Capita", "HDI"];

  static getSummaryStats(data: HDIRow[]): SummaryStats[] {
    const n = data.length;
    const results: SummaryStats[] = [];

    for (const col of this.numericColumns) {
      const vals = data.map(row => (row as any)[col] as number).filter(v => !isNaN(v));
      const count = vals.length;
      const missing = n - count;

      if (count === 0) {
        results.push({
          column: col, count: 0, mean: 0, std: 0, min: 0, q1: 0, median: 0, q3: 0, max: 0, missing
        });
        continue;
      }

      // Mean
      const sum = vals.reduce((s, v) => s + v, 0);
      const mean = sum / count;

      // Variance & Std
      const sumSqDiff = vals.reduce((s, v) => s + Math.pow(v - mean, 2), 0);
      const std = Math.sqrt(sumSqDiff / count);

      // Quantiles
      const sorted = [...vals].sort((a, b) => a - b);
      const min = sorted[0];
      const max = sorted[count - 1];
      
      const getPercentile = (p: number) => {
        const idx = p * (count - 1);
        const low = Math.floor(idx);
        const high = Math.ceil(idx);
        if (low === high) return sorted[low];
        return sorted[low] + (idx - low) * (sorted[high] - sorted[low]);
      };

      const q1 = getPercentile(0.25);
      const median = getPercentile(0.50);
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

  static getCorrelationMatrix(data: HDIRow[]): CorrelationCell[] {
    const matrix: CorrelationCell[] = [];
    const cols = this.numericColumns;

    const columnStats = cols.map(col => {
      const vals = data.map(row => (row as any)[col] as number);
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

  static getHistogramBins(data: HDIRow[], col: string, numBins = 10): HistogramBin[] {
    const vals = data.map(row => (row as any)[col] as number).filter(v => !isNaN(v));
    if (vals.length === 0) return [];

    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const range = max - min;
    const binWidth = range === 0 ? 1 : range / numBins;

    const bins: HistogramBin[] = Array.from({ length: numBins }, (_, i) => {
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

  static getBoxPlotStats(data: HDIRow[]): BoxPlotStats[] {
    const results: BoxPlotStats[] = [];

    for (const col of this.numericColumns) {
      const vals = data.map(row => (row as any)[col] as number).filter(v => !isNaN(v));
      if (vals.length === 0) continue;

      const sorted = [...vals].sort((a, b) => a - b);
      const count = sorted.length;

      const getPercentile = (p: number) => {
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

      // Find real non-outlier min and max (adjacent values)
      const nonOutliers = sorted.filter(v => v >= lowerFence && v <= upperFence);
      const min = nonOutliers.length > 0 ? nonOutliers[0] : q1;
      const max = nonOutliers.length > 0 ? nonOutliers[nonOutliers.length - 1] : q3;

      const outliers = sorted.filter(v => v < lowerFence || v > upperFence);

      results.push({
        column: col.replace(/_/g, " "),
        min: parseFloat(min.toFixed(2)),
        q1: parseFloat(q1.toFixed(2)),
        median: parseFloat(median.toFixed(2)),
        q3: parseFloat(q3.toFixed(2)),
        max: parseFloat(max.toFixed(2)),
        outliers: outliers.map(v => parseFloat(v.toFixed(2)))
      });
    }

    return results;
  }
}
