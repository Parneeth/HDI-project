import React, { useState, useEffect } from "react";
import { 
  Award, BookOpen, Heart, DollarSign, Activity, TrendingUp, 
  Compass, Brain, Play, RotateCcw, FileText, Database, Search, 
  ArrowUpDown, CheckCircle, AlertCircle, ChevronRight, Sparkles, 
  Cpu, FileSpreadsheet, Info, Check, RefreshCw
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";

// Types
interface HDIRow {
  Country: string;
  Life_Expectancy: number;
  Expected_Schooling: number;
  Mean_Schooling: number;
  GNI_Per_Capita: number;
  HDI: number;
}

interface SummaryStats {
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

interface CorrelationCell {
  x: string;
  y: string;
  value: number;
}

interface HistogramBin {
  binStart: number;
  binEnd: number;
  label: string;
  count: number;
}

interface BoxPlotStats {
  column: string;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  outliers: number[];
}

interface ScatterPoint {
  country: string;
  lifeExpectancy: number;
  expectedSchooling: number;
  meanSchooling: number;
  gniPerCapita: number;
  hdi: number;
}

interface ModelMetrics {
  r2: number;
  mae: number;
  mse: number;
  rmse: number;
}

interface TrainingResult {
  modelName: string;
  metrics: ModelMetrics;
  predictions: number[];
  residuals: number[];
}

interface FeatureImportance {
  feature: string;
  importance: number;
}

// Scenarios for quick prediction (Module 10)
interface PredictionScenario {
  id: string;
  name: string;
  description: string;
  country: string;
  lifeExpectancy: number;
  expectedSchooling: number;
  meanSchooling: number;
  gniPerCapita: number;
}

const SCENARIOS: PredictionScenario[] = [
  {
    id: "scenario-1",
    name: "Scenario 1: High Development",
    description: "Highly advanced nation with high income, robust education system, and excellent healthcare infrastructure.",
    country: "Valkyria",
    lifeExpectancy: 83.5,
    expectedSchooling: 18.5,
    meanSchooling: 13.5,
    gniPerCapita: 62000
  },
  {
    id: "scenario-2",
    name: "Scenario 2: Medium Development",
    description: "Developing transition economy with average education access and improving clinical health facilities.",
    country: "Midgard",
    lifeExpectancy: 68.5,
    expectedSchooling: 12.5,
    meanSchooling: 7.2,
    gniPerCapita: 8500
  },
  {
    id: "scenario-3",
    name: "Scenario 3: Low Development",
    description: "Least developed state with severe structural deficits, low social funding, and pressing health challenges.",
    country: "Niflheim",
    lifeExpectancy: 54.5,
    expectedSchooling: 7.5,
    meanSchooling: 3.2,
    gniPerCapita: 1100
  }
];

export default function App() {
  // Global View State
  const [activeTab, setActiveTab] = useState<"home" | "data" | "trainer" | "predictor">("home");
  const [loading, setLoading] = useState(true);

  // Backend Data Cache
  const [datasetInfo, setDatasetInfo] = useState<{
    shape: number[];
    columns: string[];
    duplicates: number;
    summary: SummaryStats[];
    rows: HDIRow[];
    totalRows: number;
  } | null>(null);

  const [edaData, setEdaData] = useState<{
    correlations: CorrelationCell[];
    boxplots: BoxPlotStats[];
    histograms: { [key: string]: HistogramBin[] };
    scatterData: ScatterPoint[];
  } | null>(null);

  const [trainInfo, setTrainInfo] = useState<{
    bestModel: string;
    metrics: { [key: string]: TrainingResult };
    featureImportance: FeatureImportance[];
    testSetCount: number;
    serialized: boolean;
  } | null>(null);

  // Interactive UI Filtering/Sorting
  const [datasetSearch, setDatasetSearch] = useState("");
  const [sortCol, setSortCol] = useState<keyof HDIRow | "">("");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  
  // EDA Chart State
  const [edaSelectedFeature, setEdaSelectedFeature] = useState<string>("Life_Expectancy");
  const [hoveredDataPoint, setHoveredDataPoint] = useState<any | null>(null);

  // Prediction Form State
  const [predForm, setPredForm] = useState({
    country: "",
    lifeExpectancy: "",
    expectedSchooling: "",
    meanSchooling: "",
    gniPerCapita: ""
  });
  const [predErrors, setPredErrors] = useState<any>({});
  const [isPredicting, setIsPredicting] = useState(false);
  const [predResult, setPredResult] = useState<any | null>(null);

  // Model Retraining State
  const [isTraining, setIsTraining] = useState(false);
  const [trainMessage, setTrainMessage] = useState<string | null>(null);

  // Load initial backend database info
  useEffect(() => {
    async function fetchInitialData() {
      try {
        setLoading(true);
        const [dsRes, edaRes] = await Promise.all([
          fetch("/api/dataset"),
          fetch("/api/eda")
        ]);

        if (dsRes.ok) {
          const dsData = await dsRes.json();
          setDatasetInfo(dsData);
        }
        if (edaRes.ok) {
          const edaJson = await edaRes.json();
          setEdaData(edaJson);
        }

        // Fetch trained model metrics
        const trainRes = await fetch("/api/train", { method: "POST" });
        if (trainRes.ok) {
          const tData = await trainRes.json();
          setTrainInfo(tData);
        }
      } catch (err) {
        console.error("Error loading application backend data:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchInitialData();
  }, []);

  // Form Validation
  const validateForm = () => {
    const errors: any = {};
    if (!predForm.country || predForm.country.trim() === "") {
      errors.country = "Country name is required.";
    }
    const le = parseFloat(predForm.lifeExpectancy);
    if (isNaN(le) || le < 20 || le > 100) {
      errors.lifeExpectancy = "Must be between 20 and 100 years.";
    }
    const es = parseFloat(predForm.expectedSchooling);
    if (isNaN(es) || es < 0 || es > 30) {
      errors.expectedSchooling = "Must be between 0 and 30 years.";
    }
    const ms = parseFloat(predForm.meanSchooling);
    if (isNaN(ms) || ms < 0 || ms > 30) {
      errors.meanSchooling = "Must be between 0 and 30 years.";
    }
    const gni = parseFloat(predForm.gniPerCapita);
    if (isNaN(gni) || gni < 100 || gni > 200000) {
      errors.gniPerCapita = "Must be between $100 and $200,000.";
    }
    setPredErrors(errors);
    return Object.keys(errors).length === 0;
  };

  // Run Prediction
  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateForm()) return;

    try {
      setIsPredicting(true);
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(predForm)
      });

      if (res.ok) {
        const result = await res.json();
        setPredResult(result);
        // Stagger visual scroll to prediction result
        setTimeout(() => {
          document.getElementById("prediction-result-panel")?.scrollIntoView({ behavior: "smooth" });
        }, 150);
      } else {
        const errObj = await res.json();
        setPredErrors({ form: errObj.error || "Prediction failed." });
      }
    } catch (err) {
      setPredErrors({ form: "Error connecting to the prediction server." });
    } finally {
      setIsPredicting(false);
    }
  };

  // Pre-fill a Scenario (Module 10)
  const applyScenario = (sc: PredictionScenario) => {
    setPredForm({
      country: sc.country,
      lifeExpectancy: sc.lifeExpectancy.toString(),
      expectedSchooling: sc.expectedSchooling.toString(),
      meanSchooling: sc.meanSchooling.toString(),
      gniPerCapita: sc.gniPerCapita.toString()
    });
    setPredErrors({});
    setPredResult(null);
  };

  // Run Retraining
  const handleRetrain = async () => {
    try {
      setIsTraining(true);
      setTrainMessage("Training and tuning all 4 models on bootstrapped dataset splits...");
      const res = await fetch("/api/train", { method: "POST" });
      if (res.ok) {
        const tData = await res.json();
        setTrainInfo(tData);
        setTrainMessage(`Success! Model serialized. Best model selected: ${tData.bestModel}`);
        setTimeout(() => setTrainMessage(null), 5000);
      } else {
        setTrainMessage("Error: Failed to retrain machine learning regressors.");
      }
    } catch (err) {
      setTrainMessage("Connection failure during training optimization.");
    } finally {
      setIsTraining(false);
    }
  };

  // Reset predictor form
  const resetForm = () => {
    setPredForm({
      country: "",
      lifeExpectancy: "",
      expectedSchooling: "",
      meanSchooling: "",
      gniPerCapita: ""
    });
    setPredErrors({});
    setPredResult(null);
  };

  // Sorted list for Country Explorer (Module 1)
  const getProcessedCountries = () => {
    if (!datasetInfo) return [];
    let items = [...datasetInfo.rows];

    // Search
    if (datasetSearch.trim() !== "") {
      items = items.filter(r => r.Country.toLowerCase().includes(datasetSearch.toLowerCase()));
    }

    // Sort
    if (sortCol !== "") {
      items.sort((a, b) => {
        const valA = a[sortCol];
        const valB = b[sortCol];
        if (typeof valA === "string" && typeof valB === "string") {
          return sortDir === "asc" ? valA.localeCompare(valB) : valB.localeCompare(valA);
        } else {
          return sortDir === "asc" ? (valA as number) - (valB as number) : (valB as number) - (valA as number);
        }
      });
    }

    return items;
  };

  const handleSort = (col: keyof HDIRow) => {
    if (sortCol === col) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortCol(col);
      setSortDir("asc");
    }
  };

  // Markdown rendering helper
  const renderMarkdown = (md: string) => {
    const lines = md.split("\n");
    return lines.map((line, idx) => {
      let text = line.trim();
      if (text.startsWith("###")) {
        return <h3 id={`md-h3-${idx}`} key={idx} className="text-lg font-bold text-slate-800 mt-5 mb-2 flex items-center gap-2"><Sparkles className="w-4 h-4 text-indigo-500" /> {text.replace("###", "").replace(/\*/g, "").trim()}</h3>;
      }
      if (text.startsWith("##")) {
        return <h2 id={`md-h2-${idx}`} key={idx} className="text-xl font-extrabold text-slate-900 mt-6 mb-3 border-b border-slate-100 pb-1">{text.replace("##", "").replace(/\*/g, "").trim()}</h2>;
      }
      if (text.startsWith("#")) {
        return <h1 id={`md-h1-${idx}`} key={idx} className="text-2xl font-black text-slate-900 mt-8 mb-4">{text.replace("#", "").replace(/\*/g, "").trim()}</h1>;
      }
      if (text.startsWith("* **") || text.startsWith("- **")) {
        const cleaned = text.replace(/^[\*\-]\s*\*\*/, "").replace(/\*\*/, "");
        const parts = cleaned.split(":");
        return (
          <li id={`md-li-${idx}`} key={idx} className="ml-5 list-disc text-slate-600 mb-2 leading-relaxed text-sm md:text-base">
            <strong className="text-slate-800 font-semibold">{parts[0]}</strong>{parts[1] ? `:${parts[1]}` : ""}
          </li>
        );
      }
      if (text.startsWith("*") || text.startsWith("-")) {
        return <li id={`md-li-plain-${idx}`} key={idx} className="ml-5 list-disc text-slate-600 mb-1 leading-relaxed text-sm md:text-base">{text.substring(1).trim()}</li>;
      }
      if (text.match(/^\d+\./)) {
        const cleaned = text.replace(/^\d+\.\s*/, "");
        if (cleaned.startsWith("**")) {
          const boldPart = cleaned.match(/^\*\*(.*?)\*\*/)?.[1] || "";
          const restPart = cleaned.replace(/^\*\*(.*?)\*\*\:?/, "");
          return (
            <div id={`md-num-${idx}`} key={idx} className="ml-2 pl-4 border-l-2 border-indigo-200 my-4 py-1">
              <p className="text-slate-700 leading-relaxed text-sm md:text-base font-normal">
                <span className="font-bold text-indigo-950 mr-1">{boldPart}</span> {restPart}
              </p>
            </div>
          );
        }
        return <div id={`md-num-plain-${idx}`} key={idx} className="ml-2 pl-4 border-l-2 border-slate-200 my-3 py-1"><p className="text-slate-600 leading-relaxed text-sm md:text-base">{cleaned}</p></div>;
      }
      if (text === "") {
        return <div key={idx} className="h-2"></div>;
      }
      
      let parts: any[] = [text];
      if (text.includes("**")) {
        const splitText = text.split("**");
        parts = splitText.map((part, pIdx) => pIdx % 2 === 1 ? <strong key={pIdx} className="font-bold text-slate-900">{part}</strong> : part);
        return <p id={`md-p-${idx}`} key={idx} className="text-slate-600 mb-3 leading-relaxed text-sm md:text-base">{parts}</p>;
      }

      return <p id={`md-p-plain-${idx}`} key={idx} className="text-slate-600 mb-3 leading-relaxed text-sm md:text-base">{text}</p>;
    });
  };

  // Render SVG Histogram (Module 2)
  const renderHistogram = () => {
    if (!edaData || !edaData.histograms[edaSelectedFeature]) return null;
    const bins = edaData.histograms[edaSelectedFeature];
    const maxCount = Math.max(...bins.map(b => b.count));
    
    const width = 600;
    const height = 280;
    const padding = 40;
    const graphWidth = width - 2 * padding;
    const graphHeight = height - 2 * padding;

    return (
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto bg-slate-50 rounded-xl border border-slate-100 p-2">
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((r, i) => (
          <line 
            key={i}
            x1={padding}
            y1={padding + graphHeight * (1 - r)}
            x2={width - padding}
            y2={padding + graphHeight * (1 - r)}
            className="stroke-slate-200 stroke-1 stroke-dasharray-[4,4]"
            strokeDasharray="4 4"
          />
        ))}

        {/* Bins / Bars */}
        {bins.map((bin, i) => {
          const barWidth = graphWidth / bins.length - 6;
          const barHeight = maxCount > 0 ? (bin.count / maxCount) * graphHeight : 0;
          const x = padding + i * (graphWidth / bins.length) + 3;
          const y = padding + graphHeight - barHeight;

          return (
            <g key={i} className="group cursor-pointer">
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={barHeight}
                className="fill-indigo-500 hover:fill-indigo-600 transition-colors duration-150 rx-2"
                rx={4}
                onMouseEnter={() => setHoveredDataPoint({
                  type: "histogram",
                  label: bin.label,
                  count: bin.count,
                  x: x + barWidth / 2,
                  y: y - 10
                })}
                onMouseLeave={() => setHoveredDataPoint(null)}
              />
              <text
                x={x + barWidth / 2}
                y={height - padding + 15}
                className="text-[9px] fill-slate-500 font-mono text-anchor-middle text-center"
                textAnchor="middle"
              >
                {bin.binStart.toFixed(0)}
              </text>
            </g>
          );
        })}

        {/* Labels */}
        <text x={width / 2} y={height - 5} className="text-[10px] font-sans fill-slate-500 text-anchor-middle text-center" textAnchor="middle">
          {edaSelectedFeature.replace(/_/g, " ")} Interval Ranges
        </text>
        <text x={10} y={height / 2} className="text-[10px] font-sans fill-slate-500 transform -rotate-90 origin-center text-anchor-middle" textAnchor="middle" transform={`rotate(-90 12 ${height / 2})`}>
          Frequency (Count)
        </text>
      </svg>
    );
  };

  // Render SVG Scatter Plot (Module 2)
  const renderScatterPlot = () => {
    if (!edaData) return null;
    const featureKey = edaSelectedFeature;
    const data = edaData.scatterData;

    let xAttr = "lifeExpectancy";
    if (featureKey === "Expected_Schooling") xAttr = "expectedSchooling";
    if (featureKey === "Mean_Schooling") xAttr = "meanSchooling";
    if (featureKey === "GNI_Per_Capita") xAttr = "gniPerCapita";
    if (featureKey === "HDI") xAttr = "hdi";

    const xVals = data.map(d => (d as any)[xAttr] as number);
    const yVals = data.map(d => d.hdi);

    const xMin = Math.min(...xVals) * 0.95;
    const xMax = Math.max(...xVals) * 1.05;
    const yMin = 0.3; // Minimum HDI baseline
    const yMax = 1.0;

    const width = 600;
    const height = 280;
    const padding = 45;
    const graphWidth = width - 2 * padding;
    const graphHeight = height - 2 * padding;

    const getX = (val: number) => padding + ((val - xMin) / (xMax - xMin)) * graphWidth;
    const getY = (val: number) => padding + graphHeight - ((val - yMin) / (yMax - yMin)) * graphHeight;

    return (
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto bg-slate-50 rounded-xl border border-slate-100 p-2">
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map((r, i) => {
          const hdiVal = yMin + r * (yMax - yMin);
          return (
            <line 
              key={i}
              x1={padding}
              y1={getY(hdiVal)}
              x2={width - padding}
              y2={getY(hdiVal)}
              className="stroke-slate-200 stroke-1 stroke-dasharray-[4,4]"
              strokeDasharray="4 4"
            />
          );
        })}

        {/* Axes */}
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} className="stroke-slate-300 stroke-1" />
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} className="stroke-slate-300 stroke-1" />

        {/* Data points */}
        {data.map((pt, i) => {
          const cx = getX((pt as any)[xAttr]);
          const cy = getY(pt.hdi);

          // Color coded based on HDI
          let ptColor = "fill-rose-500 hover:fill-rose-600";
          if (pt.hdi >= 0.8) ptColor = "fill-indigo-600 hover:fill-indigo-700";
          else if (pt.hdi >= 0.7) ptColor = "fill-teal-500 hover:fill-teal-600";
          else if (pt.hdi >= 0.55) ptColor = "fill-amber-500 hover:fill-amber-600";

          return (
            <circle
              key={i}
              cx={cx}
              cy={cy}
              r={4}
              className={`${ptColor} transition-transform duration-100 hover:scale-[2] cursor-pointer`}
              onMouseEnter={() => setHoveredDataPoint({
                type: "scatter",
                country: pt.country,
                xVal: (pt as any)[xAttr],
                yVal: pt.hdi,
                x: cx,
                y: cy - 10
              })}
              onMouseLeave={() => setHoveredDataPoint(null)}
            />
          );
        })}

        {/* Axis Labels */}
        <text x={width / 2} y={height - 5} className="text-[10px] font-sans fill-slate-500 text-anchor-middle text-center" textAnchor="middle">
          {featureKey.replace(/_/g, " ")} Value
        </text>
        <text x={10} y={height / 2} className="text-[10px] font-sans fill-slate-500 transform -rotate-90 origin-center text-anchor-middle" textAnchor="middle" transform={`rotate(-90 12 ${height / 2})`}>
          Actual HDI Score
        </text>
      </svg>
    );
  };

  // Render SVG Heatmap (Module 2)
  const renderHeatmap = () => {
    if (!edaData) return null;
    const cells = edaData.correlations;
    const labels = ["Life Expectancy", "Expected Schooling", "Mean Schooling", "GNI Per Capita", "HDI"];

    const width = 600;
    const height = 300;
    const padding = 95;
    const cellSize = (width - padding - 20) / labels.length;

    return (
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto bg-slate-50 rounded-xl border border-slate-100 p-2">
        {/* Draw Cells */}
        {cells.map((cell, idx) => {
          const xIdx = labels.indexOf(cell.x);
          const yIdx = labels.indexOf(cell.y);

          const x = padding + xIdx * cellSize;
          const y = 20 + yIdx * cellSize;

          // Color scale: blue positive, white zero, red negative (mostly highly positive for HDI)
          const val = cell.value;
          const r = val > 0 ? 99 : 244;
          const g = val > 0 ? Math.floor(102 + (1 - val) * 140) : Math.floor(63 + (1 + val) * 180);
          const b = val > 0 ? 241 : 94;

          return (
            <g key={idx} className="group cursor-pointer">
              <rect
                x={x}
                y={y}
                width={cellSize - 1}
                height={cellSize - 1}
                fill={`rgb(${r}, ${g}, ${b})`}
                className="transition-opacity duration-150 hover:opacity-85"
                onMouseEnter={() => setHoveredDataPoint({
                  type: "heatmap",
                  xLabel: cell.x,
                  yLabel: cell.y,
                  val: cell.value,
                  x: x + cellSize / 2,
                  y: y - 10
                })}
                onMouseLeave={() => setHoveredDataPoint(null)}
              />
              <text
                x={x + cellSize / 2}
                y={y + cellSize / 2 + 4}
                className={`text-[9px] font-mono font-bold text-center text-anchor-middle`}
                textAnchor="middle"
                fill={Math.abs(val) > 0.6 ? "#ffffff" : "#1e293b"}
              >
                {val.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* Labels X */}
        {labels.map((lbl, idx) => (
          <text
            key={`x-${idx}`}
            x={padding + idx * cellSize + cellSize / 2}
            y={height - 2}
            className="text-[8px] font-sans font-medium fill-slate-600 text-anchor-middle text-center"
            textAnchor="middle"
          >
            {lbl}
          </text>
        ))}

        {/* Labels Y */}
        {labels.map((lbl, idx) => (
          <text
            key={`y-${idx}`}
            x={padding - 10}
            y={20 + idx * cellSize + cellSize / 2 + 3}
            className="text-[8px] font-sans font-medium fill-slate-600 text-right"
            textAnchor="end"
          >
            {lbl}
          </text>
        ))}
      </svg>
    );
  };

  // Render SVG Feature Importance (Module 4)
  const renderFeatureImportance = () => {
    if (!trainInfo) return null;
    const importance = trainInfo.featureImportance;
    const height = 180;
    const width = 500;
    const padding = 100;
    const graphWidth = width - padding - 20;
    const barHeight = 22;

    return (
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto bg-slate-50 rounded-xl border border-slate-100 p-2">
        {importance.map((item, idx) => {
          const y = 20 + idx * (barHeight + 15);
          const widthVal = item.importance * graphWidth;

          return (
            <g key={idx}>
              <text x={padding - 10} y={y + 15} className="text-[10px] font-sans font-bold fill-slate-700 text-right" textAnchor="end">
                {item.feature.replace(/_/g, " ")}
              </text>
              <rect
                x={padding}
                y={y}
                width={widthVal}
                height={barHeight}
                className="fill-indigo-500 rx-2 hover:fill-indigo-600 transition-colors"
                rx={3}
              />
              <text x={padding + widthVal + 8} y={y + 15} className="text-[10px] font-mono font-bold fill-slate-800">
                {(item.importance * 100).toFixed(1)}%
              </text>
            </g>
          );
        })}
      </svg>
    );
  };

  return (
    <div id="application-container" className="h-screen w-full overflow-hidden font-sans text-slate-900 bg-slate-50 flex selection:bg-indigo-100 selection:text-indigo-900">
      
      {/* Navigation Rail */}
      <nav className="w-20 bg-slate-900 flex flex-col items-center py-6 border-r border-slate-800 shrink-0 select-none">
        {/* Upper Logo */}
        <div className="w-10 h-10 bg-indigo-500 rounded-lg flex items-center justify-center mb-10 shadow-lg shadow-indigo-500/20 shrink-0">
          <Award className="text-white w-5 h-5" />
        </div>
        
        {/* Navigation Buttons */}
        <div className="flex flex-col gap-6 w-full items-center">
          {[
            { id: "home", label: "About", icon: Compass },
            { id: "data", label: "Data Lab", icon: FileSpreadsheet },
            { id: "trainer", label: "Trainer", icon: Cpu },
            { id: "predictor", label: "Predictor", icon: Brain }
          ].map((tab) => {
            const TabIcon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                id={`tab-btn-${tab.id}`}
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`w-14 h-14 rounded-xl flex flex-col items-center justify-center transition-all border ${
                  isActive 
                    ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/30 border-indigo-400" 
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800 border-transparent"
                }`}
                title={tab.label}
              >
                <TabIcon className="w-4 h-4 shrink-0" />
                <span className="text-[9px] font-semibold mt-1 scale-90 leading-none shrink-0">{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Bottom decorative piece (User avatar) */}
        <div className="mt-auto">
          <div className="w-8 h-8 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-xs text-slate-400 font-bold font-mono">
            U
          </div>
        </div>
      </nav>

      {/* Main Workspace */}
      <main className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Top Header */}
        <header className="h-16 border-b border-slate-200 bg-white px-8 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <h1 className="text-base font-extrabold tracking-tight text-slate-900 sm:text-lg">
              Human Development Index (HDI) Prediction System
            </h1>
            <span className="bg-emerald-50 text-emerald-700 text-[10px] font-bold px-2 py-0.5 rounded-full border border-emerald-100 uppercase tracking-wider">
              V2.4 PRODUCTION
            </span>
          </div>
          
          <div className="hidden md:flex items-center gap-4 text-xs font-mono text-slate-500">
            <span>MODEL: {trainInfo?.bestModel?.toUpperCase().replace(/ /g, "_") || "RANDOM_FOREST_REGRESSOR"}</span>
            <span className="h-4 w-px bg-slate-200"></span>
            <span>ACCURACY: R² {trainInfo?.metrics[trainInfo?.bestModel || "Random Forest"]?.metrics.r2.toFixed(4) || "0.984"}</span>
          </div>
        </header>

        {/* Content Container */}
        <div className="flex-1 overflow-hidden relative">
          {loading ? (
            /* Loading Skeleton */
            <div className="h-full flex flex-col items-center justify-center gap-4 bg-slate-50">
              <RefreshCw className="w-8 h-8 text-indigo-600 animate-spin" />
              <p className="text-sm font-mono text-slate-500">Initializing human development dataset & compiling ML classifiers...</p>
            </div>
          ) : (
            <AnimatePresence mode="wait">
              
              {/* TAB 1: ABOUT / HOME */}
              {activeTab === "home" && (
                <motion.div
                  key="home"
                  initial={{ opacity: 0, y: 15 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -15 }}
                  className="h-full overflow-y-auto p-8 space-y-8 bg-slate-50"
                >
                  {/* Hero Banner Grid */}
                  <div className="bg-slate-900 text-white rounded-2xl p-8 sm:p-12 relative overflow-hidden border border-slate-800 shadow-xl">
                    <div className="absolute -right-20 -top-20 w-80 h-80 bg-indigo-500 rounded-full filter blur-[100px] opacity-15" />
                    
                    <div className="max-w-2xl relative z-10 space-y-4">
                      <div className="inline-flex items-center gap-2 px-3 py-1 bg-white/10 rounded-full border border-white/15">
                        <Sparkles className="w-3.5 h-3.5 text-indigo-300" />
                        <span className="text-[10px] sm:text-xs font-semibold tracking-wider text-indigo-100 uppercase">UN Human Development Standards</span>
                      </div>
                      <h2 className="text-3xl sm:text-5xl font-extrabold tracking-tight leading-tight">
                        Predicting Development. <br />
                        <span className="text-indigo-400">Informing Policy.</span>
                      </h2>
                      <p className="text-sm sm:text-base text-slate-300 font-normal leading-relaxed">
                        A robust, machine-learning-driven analytics web platform designed to analyze socioeconomic factors and predict the Human Development Index (HDI) scores of sovereign nations.
                      </p>
                      <div className="pt-4 flex flex-wrap gap-3">
                        <button 
                          onClick={() => setActiveTab("predictor")}
                          className="px-6 py-2.5 bg-indigo-500 hover:bg-indigo-600 text-white text-xs font-bold rounded-lg tracking-wider uppercase shadow-md transition-all"
                        >
                          Launch Predictor
                        </button>
                        <button 
                          onClick={() => setActiveTab("data")}
                          className="px-6 py-2.5 bg-white/10 hover:bg-white/15 border border-white/15 text-white text-xs font-bold rounded-lg tracking-wider uppercase transition-all"
                        >
                          Explore Raw Dataset
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Conceptual Breakdown Bento Grid */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    
                    {/* Card 1: What is HDI */}
                    <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm space-y-3">
                      <div className="p-3 bg-indigo-50 text-indigo-600 rounded-lg w-fit">
                        <BookOpen className="w-5 h-5" />
                      </div>
                      <h3 className="text-base font-bold text-slate-900">What is HDI?</h3>
                      <p className="text-xs text-slate-600 leading-relaxed">
                        The **Human Development Index (HDI)** is a statistic composite index published by the United Nations. It is designed to measure the social and economic development levels of countries globally across three fundamental pillars.
                      </p>
                    </div>

                    {/* Card 2: Objective & Target */}
                    <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm space-y-3">
                      <div className="p-3 bg-emerald-50 text-emerald-600 rounded-lg w-fit">
                        <TrendingUp className="w-5 h-5" />
                      </div>
                      <h3 className="text-base font-bold text-slate-900">Platform Objective</h3>
                      <p className="text-xs text-slate-600 leading-relaxed">
                        This system assists **analysts, researchers, students, and policy advisors** in projecting the immediate developmental impact of socioeconomic changes, highlighting critical resource bottlenecks.
                      </p>
                    </div>

                    {/* Card 3: Interactive ML */}
                    <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm space-y-3">
                      <div className="p-3 bg-amber-50 text-amber-600 rounded-lg w-fit">
                        <Cpu className="w-5 h-5" />
                      </div>
                      <h3 className="text-base font-bold text-slate-900">Ensemble Algorithms</h3>
                      <p className="text-xs text-slate-600 leading-relaxed">
                        Our prediction pipeline compares four machine learning algorithms (OLS Linear Regression, Decision Trees, Random Forests, and Gradient Boosting) to secure optimal predictive precision.
                      </p>
                    </div>

                  </div>

                  {/* Pillars of HDI Index Breakdown */}
                  <div className="bg-white rounded-xl p-6 sm:p-8 border border-slate-200 shadow-sm space-y-6">
                    <div className="space-y-1">
                      <h3 className="text-lg font-bold text-slate-900">Three Dimensions of Human Development</h3>
                      <p className="text-xs text-slate-500">The primary metrics our machine learning model uses to calculate human capability thresholds.</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
                      
                      {/* Dimension 1: Longevity */}
                      <div className="flex items-start gap-4 p-4 rounded-lg hover:bg-slate-50/50 transition-colors">
                        <div className="p-2.5 bg-rose-50 text-rose-500 rounded-lg shrink-0">
                          <Heart className="w-5 h-5" />
                        </div>
                        <div className="space-y-1">
                          <h4 className="text-xs font-bold text-slate-800">1. Longevity & Health</h4>
                          <p className="text-[11px] text-slate-500 leading-relaxed">
                            Measured via **Life Expectancy at Birth**. Reflects a country's clinical services, dietary habits, sanitary frameworks, and social safety nets.
                          </p>
                        </div>
                      </div>

                      {/* Dimension 2: Education */}
                      <div className="flex items-start gap-4 p-4 rounded-lg hover:bg-slate-50/50 transition-colors">
                        <div className="p-2.5 bg-indigo-50 text-indigo-500 rounded-lg shrink-0">
                          <BookOpen className="w-5 h-5" />
                        </div>
                        <div className="space-y-1">
                          <h4 className="text-xs font-bold text-slate-800">2. Knowledge Access</h4>
                          <p className="text-[11px] text-slate-500 leading-relaxed">
                            Measured via **Expected Schooling Years** for children and **Mean Schooling Years** for adults. Gauges secondary and tertiary capacity.
                          </p>
                        </div>
                      </div>

                      {/* Dimension 3: Income */}
                      <div className="flex items-start gap-4 p-4 rounded-lg hover:bg-slate-50/50 transition-colors">
                        <div className="p-2.5 bg-emerald-50 text-emerald-500 rounded-lg shrink-0">
                          <DollarSign className="w-5 h-5" />
                        </div>
                        <div className="space-y-1">
                          <h4 className="text-xs font-bold text-slate-800">3. Living Standards</h4>
                          <p className="text-[11px] text-slate-500 leading-relaxed">
                            Measured via **Gross National Income (GNI) Per Capita (PPP)**. Evaluates actual household domestic purchasing capacity.
                          </p>
                        </div>
                      </div>

                    </div>
                  </div>
                </motion.div>
              )}

              {/* TAB 2: DATA EXPLORER & EDA */}
              {activeTab === "data" && (
                <motion.div
                  key="data"
                  initial={{ opacity: 0, y: 15 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -15 }}
                  className="h-full overflow-y-auto p-8 grid grid-cols-1 lg:grid-cols-12 gap-8 bg-slate-50"
                >
                  {/* Left Column: Summary Stats & Raw Data Explorer (7 Cols) */}
                  <div className="lg:col-span-7 space-y-8">
                    
                    {/* Summary Statistics Table */}
                    <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm space-y-4">
                      <div className="flex items-center gap-2">
                        <Database className="w-5 h-5 text-indigo-600" />
                        <h3 className="text-sm font-bold text-slate-900">Dataset Descriptive Statistics</h3>
                      </div>
                      <div className="overflow-x-auto border border-slate-200 rounded-lg">
                        <table className="w-full text-left border-collapse">
                          <thead>
                            <tr className="bg-slate-50 border-b border-slate-200 text-[11px] font-bold text-slate-500">
                              <th className="px-4 py-2.5">Indicator</th>
                              <th className="px-4 py-2.5 text-right">Count</th>
                              <th className="px-4 py-2.5 text-right">Mean</th>
                              <th className="px-4 py-2.5 text-right">Std Dev</th>
                              <th className="px-4 py-2.5 text-right">Min</th>
                              <th className="px-4 py-2.5 text-right">Median</th>
                              <th className="px-4 py-2.5 text-right">Max</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-100 text-xs font-mono">
                            {datasetInfo?.summary.map((stat, i) => (
                              <tr key={i} className="hover:bg-slate-50/50">
                                <td className="px-4 py-2 font-sans font-bold text-slate-700">{stat.column}</td>
                                <td className="px-4 py-2 text-right text-slate-500">{stat.count}</td>
                                <td className="px-4 py-2 text-right text-slate-600">{stat.mean.toLocaleString()}</td>
                                <td className="px-4 py-2 text-right text-slate-500">{stat.std}</td>
                                <td className="px-4 py-2 text-right text-slate-500">{stat.min.toLocaleString()}</td>
                                <td className="px-4 py-2 text-right text-indigo-600 font-semibold">{stat.median.toLocaleString()}</td>
                                <td className="px-4 py-2 text-right text-slate-600">{stat.max.toLocaleString()}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <div className="flex flex-wrap items-center justify-between gap-3 text-[10px] font-mono text-slate-400">
                        <span>Columns: 6 (Country, Life_Exp, Exp_Sch, Mean_Sch, GNI, HDI)</span>
                        <div className="flex gap-4">
                          <span>Missing values: 0</span>
                          <span>Detected duplicates: {datasetInfo?.duplicates || 0}</span>
                        </div>
                      </div>
                    </div>

                    {/* Raw Data Explorer */}
                    <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm space-y-4">
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                        <div className="flex items-center gap-2">
                          <FileSpreadsheet className="w-5 h-5 text-indigo-600" />
                          <h3 className="text-sm font-bold text-slate-900">Socioeconomic Country Database</h3>
                        </div>
                        
                        {/* Search */}
                        <div className="relative max-w-xs w-full">
                          <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                          <input
                            type="text"
                            placeholder="Search countries..."
                            value={datasetSearch}
                            onChange={(e) => setDatasetSearch(e.target.value)}
                            className="w-full pl-9 pr-4 py-1.5 border border-slate-200 rounded-lg text-xs font-semibold bg-slate-50 focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:bg-white"
                          />
                        </div>
                      </div>

                      {/* Table */}
                      <div className="overflow-x-auto border border-slate-200 rounded-lg">
                        <table className="w-full text-left border-collapse">
                          <thead>
                            <tr className="bg-slate-50 border-b border-slate-200 text-[11px] font-bold text-slate-500 select-none">
                              <th onClick={() => handleSort("Country")} className="px-4 py-2.5 cursor-pointer hover:bg-slate-100/50">
                                <div className="flex items-center gap-1.5">Country <ArrowUpDown className="w-3 h-3 text-slate-400" /></div>
                              </th>
                              <th onClick={() => handleSort("Life_Expectancy")} className="px-4 py-2.5 cursor-pointer hover:bg-slate-100/50 text-right">
                                <div className="flex items-center justify-end gap-1.5">Life Expectancy <ArrowUpDown className="w-3 h-3 text-slate-400" /></div>
                              </th>
                              <th onClick={() => handleSort("Expected_Schooling")} className="px-4 py-2.5 cursor-pointer hover:bg-slate-100/50 text-right">
                                <div className="flex items-center justify-end gap-1.5">Exp Schooling <ArrowUpDown className="w-3 h-3 text-slate-400" /></div>
                              </th>
                              <th onClick={() => handleSort("Mean_Schooling")} className="px-4 py-2.5 cursor-pointer hover:bg-slate-100/50 text-right">
                                <div className="flex items-center justify-end gap-1.5">Mean Schooling <ArrowUpDown className="w-3 h-3 text-slate-400" /></div>
                              </th>
                              <th onClick={() => handleSort("GNI_Per_Capita")} className="px-4 py-2.5 cursor-pointer hover:bg-slate-100/50 text-right">
                                <div className="flex items-center justify-end gap-1.5">GNI per Capita <ArrowUpDown className="w-3 h-3 text-slate-400" /></div>
                              </th>
                              <th onClick={() => handleSort("HDI")} className="px-4 py-2.5 cursor-pointer hover:bg-slate-100/50 text-right">
                                <div className="flex items-center justify-end gap-1.5">Target HDI <ArrowUpDown className="w-3 h-3 text-slate-400" /></div>
                              </th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-100 text-xs font-mono text-slate-500">
                            {getProcessedCountries().map((row, i) => {
                              let catColor = "bg-rose-50 text-rose-700 border-rose-100";
                              if (row.HDI >= 0.8) catColor = "bg-indigo-50 text-indigo-700 border-indigo-100";
                              else if (row.HDI >= 0.7) catColor = "bg-emerald-50 text-emerald-700 border-emerald-100";
                              else if (row.HDI >= 0.55) catColor = "bg-amber-50 text-amber-700 border-amber-100";

                              return (
                                <tr key={i} className="hover:bg-slate-50/50 transition-colors">
                                  <td className="px-4 py-2 font-sans font-bold text-slate-800">{row.Country}</td>
                                  <td className="px-4 py-2 text-right">{row.Life_Expectancy.toFixed(1)}</td>
                                  <td className="px-4 py-2 text-right">{row.Expected_Schooling.toFixed(1)}</td>
                                  <td className="px-4 py-2 text-right">{row.Mean_Schooling.toFixed(1)}</td>
                                  <td className="px-4 py-2 text-right">${row.GNI_Per_Capita.toLocaleString()}</td>
                                  <td className="px-4 py-2 text-right">
                                    <span className={`px-2 py-0.5 rounded-full border text-[10px] font-bold ${catColor}`}>
                                      {row.HDI.toFixed(3)}
                                    </span>
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>

                  </div>

                  {/* Right Column: Visual EDA Charts (5 Cols) */}
                  <div className="lg:col-span-5 space-y-8">
                    
                    {/* Selector Header */}
                    <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm space-y-5 relative">
                      <div className="space-y-1">
                        <h3 className="text-sm font-bold text-slate-900">Exploratory Data Analysis</h3>
                        <p className="text-xs text-slate-400">Generate and inspect distributions, correlation matrices, and scatter plots.</p>
                      </div>

                      {/* Dropdown Selector */}
                      <div className="space-y-1">
                        <label className="text-[10px] font-bold tracking-wider text-slate-400 uppercase">Select Factor to Plot</label>
                        <select
                          value={edaSelectedFeature}
                          onChange={(e) => setEdaSelectedFeature(e.target.value)}
                          className="w-full px-3 py-1.5 bg-slate-50 border border-slate-200 rounded-lg text-xs font-semibold focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:bg-white"
                        >
                          <option value="Life_Expectancy">Life Expectancy</option>
                          <option value="Expected_Schooling">Expected Years of Schooling</option>
                          <option value="Mean_Schooling">Mean Years of Schooling</option>
                          <option value="GNI_Per_Capita">Gross National Income (PPP)</option>
                          <option value="HDI">Target Human Development Index (HDI)</option>
                        </select>
                      </div>

                      {/* Chart Tab Container */}
                      <div className="space-y-6 pt-4 border-t border-slate-100">
                        
                        {/* Plot 1: Histogram */}
                        <div className="space-y-2 relative">
                          <h4 className="text-xs font-bold tracking-wide text-slate-600 flex items-center gap-1.5">
                            <Activity className="w-3.5 h-3.5 text-indigo-500" /> Feature Distribution Interval (Histogram)
                          </h4>
                          {renderHistogram()}
                        </div>

                        {/* Plot 2: Scatter vs Target */}
                        {edaSelectedFeature !== "HDI" && (
                          <div className="space-y-2">
                            <h4 className="text-xs font-bold tracking-wide text-slate-600 flex items-center gap-1.5">
                              <TrendingUp className="w-3.5 h-3.5 text-indigo-500" /> Relationship with HDI (Scatter Plot)
                            </h4>
                            {renderScatterPlot()}
                          </div>
                        )}

                        {/* Plot 3: Pearson Heatmap */}
                        <div className="space-y-2">
                          <h4 className="text-xs font-bold tracking-wide text-slate-600 flex items-center gap-1.5">
                            <Database className="w-3.5 h-3.5 text-indigo-500" /> Correlation Matrix Heatmap
                          </h4>
                          {renderHeatmap()}
                        </div>

                      </div>

                      {/* Hover Chart Tooltip Overlay */}
                      {hoveredDataPoint && (
                        <div 
                          className="p-2.5 bg-slate-900/95 text-white text-[11px] rounded-lg absolute pointer-events-none z-50 shadow-md font-sans border border-slate-800 space-y-0.5"
                          style={{
                            left: `${hoveredDataPoint.x}px`,
                            top: `${hoveredDataPoint.y}px`
                          }}
                        >
                          {hoveredDataPoint.type === "histogram" && (
                            <>
                              <div className="font-bold">{hoveredDataPoint.label}</div>
                              <div className="text-indigo-300 font-mono">Count: {hoveredDataPoint.count} Countries</div>
                            </>
                          )}
                          {hoveredDataPoint.type === "scatter" && (
                            <>
                              <div className="font-bold text-indigo-200">{hoveredDataPoint.country}</div>
                              <div className="font-mono">Value: {hoveredDataPoint.xVal.toLocaleString()}</div>
                              <div className="font-mono text-emerald-300">HDI: {hoveredDataPoint.yVal.toFixed(3)}</div>
                            </>
                          )}
                          {hoveredDataPoint.type === "heatmap" && (
                            <>
                              <div className="font-bold text-slate-300">{hoveredDataPoint.xLabel} × {hoveredDataPoint.yLabel}</div>
                              <div className="font-mono text-indigo-300 text-sm">r = {hoveredDataPoint.val.toFixed(3)}</div>
                            </>
                          )}
                        </div>
                      )}

                    </div>

                  </div>
                </motion.div>
              )}

              {/* TAB 3: MODEL TRAINER */}
              {activeTab === "trainer" && (
                <motion.div
                  key="trainer"
                  initial={{ opacity: 0, y: 15 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -15 }}
                  className="h-full overflow-y-auto p-8 space-y-8 bg-slate-50"
                >
                  
                  {/* Training Actions Card */}
                  <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
                    <div className="space-y-1 max-w-xl">
                      <h3 className="text-base font-bold text-slate-900">Machine Learning Training Laboratory</h3>
                      <p className="text-xs text-slate-500 leading-relaxed">
                        Train, optimize, and cross-evaluate 4 distinct regression architectures using a 75/25 split ratio. The model serialized in the backend updates automatically upon completion.
                      </p>
                    </div>
                    
                    <div className="flex items-center gap-3 w-full md:w-auto shrink-0">
                      <button
                        onClick={handleRetrain}
                        disabled={isTraining}
                        className="w-full md:w-auto px-6 py-2.5 bg-slate-900 hover:bg-slate-850 disabled:bg-slate-300 text-white font-bold rounded-lg text-xs tracking-wider uppercase flex items-center justify-center gap-2 shadow-sm transition-colors"
                      >
                        {isTraining ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                        {isTraining ? "Optimizing..." : "Run ML Pipeline"}
                      </button>
                    </div>
                  </div>

                  {/* Training Status Message */}
                  {trainMessage && (
                    <div className="p-4 bg-indigo-50 border border-indigo-100 text-indigo-900 rounded-xl text-xs font-mono flex items-center gap-3">
                      <Activity className="w-4 h-4 text-indigo-500 animate-pulse" />
                      <span>{trainMessage}</span>
                    </div>
                  )}

                  {/* Models Metrics Comparison Grid */}
                  <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                    
                    {/* Model Performance Comparison (8 Cols) */}
                    <div className="lg:col-span-8 bg-white rounded-xl p-6 border border-slate-200 shadow-sm space-y-5">
                      <div className="flex items-center gap-2">
                        <Cpu className="w-5 h-5 text-indigo-600" />
                        <h3 className="text-sm font-bold text-slate-900">Ensemble Regressor Performance Matrix</h3>
                      </div>

                      <div className="overflow-x-auto border border-slate-200 rounded-lg">
                        <table className="w-full text-left border-collapse">
                          <thead>
                            <tr className="bg-slate-50 border-b border-slate-200 text-[11px] font-bold text-slate-500">
                              <th className="px-4 py-3">Regression Model</th>
                              <th className="px-4 py-3 text-right">R² Score (Precision)</th>
                              <th className="px-4 py-3 text-right">Mean Absolute Error (MAE)</th>
                              <th className="px-4 py-3 text-right">Mean Squared Error (MSE)</th>
                              <th className="px-4 py-3 text-right">Root MSE (RMSE)</th>
                              <th className="px-4 py-3 text-center">Status</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-100 text-xs font-mono text-slate-600">
                            {trainInfo && (Object.entries(trainInfo.metrics) as [string, TrainingResult][]).map(([name, res]) => {
                              const isBest = name === trainInfo.bestModel;
                              return (
                                <tr key={name} className={`hover:bg-slate-50/50 ${isBest ? "bg-indigo-50/30 font-semibold" : ""}`}>
                                  <td className="px-4 py-3 font-sans font-bold text-slate-800 flex items-center gap-2">
                                    {isBest && <Award className="w-4 h-4 text-indigo-500" />}
                                    {name}
                                  </td>
                                  <td className="px-4 py-3 text-right font-bold text-indigo-600">{res.metrics.r2.toFixed(4)}</td>
                                  <td className="px-4 py-3 text-right">{res.metrics.mae.toFixed(5)}</td>
                                  <td className="px-4 py-3 text-right">{res.metrics.mse.toFixed(5)}</td>
                                  <td className="px-4 py-3 text-right">{res.metrics.rmse.toFixed(5)}</td>
                                  <td className="px-4 py-3 text-center font-sans">
                                    {isBest ? (
                                      <span className="px-2.5 py-0.5 bg-indigo-100 text-indigo-700 border border-indigo-200 text-[10px] font-bold rounded-full uppercase tracking-wider">
                                        Best Selected
                                      </span>
                                    ) : (
                                      <span className="px-2.5 py-0.5 bg-slate-100 text-slate-400 text-[10px] rounded-full uppercase tracking-wider">
                                        Evaluated
                                      </span>
                                    )}
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>

                      {/* Best Model Summary Card */}
                      {trainInfo && (
                        <div className="p-5 bg-slate-900 rounded-xl text-white flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border border-slate-800">
                          <div className="space-y-1">
                            <span className="text-[9px] font-bold uppercase tracking-widest text-indigo-300 font-mono">Auto-Tuned Best Estimator</span>
                            <h4 className="text-base font-bold">{trainInfo.bestModel}</h4>
                            <p className="text-xs text-slate-300">Achieved a testing R² score of {(trainInfo.metrics[trainInfo.bestModel].metrics.r2 * 100).toFixed(2)}% explaining variance in global socioeconomic conditions.</p>
                          </div>
                          <div className="px-4 py-2.5 bg-slate-800 rounded-lg text-center shrink-0 border border-slate-700">
                            <span className="block text-[10px] text-slate-400 uppercase tracking-wider font-mono">Residual variance (MSE)</span>
                            <span className="text-base font-mono font-black text-indigo-400">{trainInfo.metrics[trainInfo.bestModel].metrics.mse.toFixed(6)}</span>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Feature Importance Panel (4 Cols) */}
                    <div className="lg:col-span-4 bg-white rounded-xl p-6 border border-slate-200 shadow-sm space-y-4">
                      <div className="flex items-center gap-2">
                        <TrendingUp className="w-5 h-5 text-indigo-600" />
                        <h3 className="text-sm font-bold text-slate-900">Feature Importance</h3>
                      </div>
                      <p className="text-xs text-slate-500 leading-relaxed">
                        Relative importance representing which social indicator contributes the most predictive variance to the best-performing ensemble regressor.
                      </p>
                      {renderFeatureImportance()}
                    </div>

                  </div>

                </motion.div>
              )}

              {/* TAB 4: PREDICTOR FORM & RESULTS */}
              {activeTab === "predictor" && (
                <motion.div
                  key="predictor"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="h-full grid grid-cols-12 overflow-hidden bg-slate-50"
                >
                  {/* Left Column: Input Panel (4 Cols) */}
                  <section className="col-span-12 lg:col-span-4 bg-white border-r border-slate-200 p-8 flex flex-col h-full overflow-y-auto">
                    <div className="mb-6">
                      <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">Prediction Engine</h2>
                      <p className="text-xs text-slate-600 leading-relaxed">Input socioeconomic parameters or load preset scenario templates to forecast developmental status.</p>
                    </div>

                    {/* Predefined Scenarios */}
                    <div className="space-y-2 mb-6">
                      <label className="text-[10px] font-bold tracking-wider text-slate-400 uppercase">Apply Scenario Template</label>
                      <div className="grid grid-cols-1 gap-2">
                        {SCENARIOS.map((sc) => (
                          <button
                            key={sc.id}
                            type="button"
                            onClick={() => applyScenario(sc)}
                            className="text-left p-3 border border-slate-200 rounded-lg hover:border-indigo-400 hover:bg-slate-50/50 transition-all flex items-start gap-3 group"
                          >
                            <Compass className="w-4 h-4 text-slate-400 group-hover:text-indigo-500 mt-0.5 shrink-0" />
                            <div>
                              <span className="block text-xs font-bold text-slate-700 group-hover:text-indigo-950">{sc.name}</span>
                              <span className="block text-[10px] text-slate-400 leading-normal mt-0.5">{sc.description}</span>
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="border-t border-slate-100 pt-5 flex-1">
                      <form onSubmit={handlePredict} className="space-y-4">
                        
                        {/* Form Error */}
                        {predErrors.form && (
                          <div className="p-3 bg-rose-50 border border-rose-100 text-rose-800 text-xs rounded-lg flex items-center gap-2 font-semibold">
                            <AlertCircle className="w-4 h-4 text-rose-500" />
                            <span>{predErrors.form}</span>
                          </div>
                        )}

                        {/* 1. Country Name */}
                        <div>
                          <label className="block text-xs font-semibold mb-2 text-slate-700">Country Name</label>
                          <input
                            type="text"
                            value={predForm.country}
                            onChange={(e) => setPredForm({ ...predForm, country: e.target.value })}
                            placeholder="e.g. Solaria"
                            className="w-full px-3 py-2 border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:bg-white bg-slate-50 font-sans"
                          />
                          {predErrors.country && <span className="text-[10px] text-rose-500 font-bold mt-1 block">{predErrors.country}</span>}
                        </div>

                        {/* 2. Life Expectancy */}
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <label className="block text-xs font-semibold text-slate-700">Life Expectancy (Years)</label>
                            <span className="text-[9px] text-slate-400 font-mono">[20 - 100]</span>
                          </div>
                          <input
                            type="number"
                            step="0.1"
                            value={predForm.lifeExpectancy}
                            onChange={(e) => setPredForm({ ...predForm, lifeExpectancy: e.target.value })}
                            placeholder="e.g. 75.4"
                            className="w-full px-3 py-2 border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:bg-white bg-slate-50 font-mono"
                          />
                          {predErrors.lifeExpectancy && <span className="text-[10px] text-rose-500 font-bold mt-1 block">{predErrors.lifeExpectancy}</span>}
                        </div>

                        {/* 3. Expected Years of Schooling */}
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <label className="block text-xs font-semibold text-slate-700">Expected Years of Schooling</label>
                            <span className="text-[9px] text-slate-400 font-mono">[0 - 30]</span>
                          </div>
                          <input
                            type="number"
                            step="0.1"
                            value={predForm.expectedSchooling}
                            onChange={(e) => setPredForm({ ...predForm, expectedSchooling: e.target.value })}
                            placeholder="e.g. 14.5"
                            className="w-full px-3 py-2 border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:bg-white bg-slate-50 font-mono"
                          />
                          {predErrors.expectedSchooling && <span className="text-[10px] text-rose-500 font-bold mt-1 block">{predErrors.expectedSchooling}</span>}
                        </div>

                        {/* 4. Mean Years of Schooling */}
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <label className="block text-xs font-semibold text-slate-700">Mean Years of Schooling</label>
                            <span className="text-[9px] text-slate-400 font-mono">[0 - 30]</span>
                          </div>
                          <input
                            type="number"
                            step="0.1"
                            value={predForm.meanSchooling}
                            onChange={(e) => setPredForm({ ...predForm, meanSchooling: e.target.value })}
                            placeholder="e.g. 10.2"
                            className="w-full px-3 py-2 border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:bg-white bg-slate-50 font-mono"
                          />
                          {predErrors.meanSchooling && <span className="text-[10px] text-rose-500 font-bold mt-1 block">{predErrors.meanSchooling}</span>}
                        </div>

                        {/* 5. GNI Per Capita */}
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <label className="block text-xs font-semibold text-slate-700">GNI per Capita (PPP $)</label>
                            <span className="text-[9px] text-slate-400 font-mono">[$100 - $200,000]</span>
                          </div>
                          <input
                            type="number"
                            value={predForm.gniPerCapita}
                            onChange={(e) => setPredForm({ ...predForm, gniPerCapita: e.target.value })}
                            placeholder="e.g. 18500"
                            className="w-full px-3 py-2 border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:bg-white bg-slate-50 font-mono"
                          />
                          {predErrors.gniPerCapita && <span className="text-[10px] text-rose-500 font-bold mt-1 block">{predErrors.gniPerCapita}</span>}
                        </div>

                        {/* Form Buttons */}
                        <div className="pt-4 space-y-2">
                          <button
                            type="submit"
                            disabled={isPredicting}
                            className="w-full bg-slate-900 text-white font-bold py-3 rounded-lg shadow-md hover:bg-slate-800 transition-colors flex items-center justify-center gap-2 text-xs tracking-wider uppercase"
                          >
                            {isPredicting ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                            <span>{isPredicting ? "RUNNING ENGINE..." : "RUN PREDICTION ENGINE"}</span>
                          </button>
                          <button
                            type="button"
                            onClick={resetForm}
                            className="w-full text-slate-500 text-xs font-bold py-2 border border-transparent hover:border-slate-200 rounded transition-all"
                          >
                            RESET TO DEFAULTS
                          </button>
                        </div>

                      </form>
                    </div>

                    <div className="mt-auto pt-6 border-t border-slate-100 shrink-0">
                      <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono">
                        <span>SCALER: STANDARD_SCALER</span>
                        <span>PICKLE_V4</span>
                      </div>
                    </div>
                  </section>

                  {/* Right Column: Prediction Results Dashboard (8 Cols) */}
                  <section id="prediction-result-panel" className="col-span-12 lg:col-span-8 p-10 flex flex-col h-full overflow-y-auto bg-slate-50">
                    {predResult ? (
                      <motion.div
                        initial={{ opacity: 0, scale: 0.98 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="space-y-8"
                      >
                        <div className="flex justify-between items-start">
                          <div>
                            <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-1">Classification Result</h2>
                            <p className="text-2xl font-light text-slate-800">Predicted Analysis: <span className="font-bold text-indigo-600 underline">{predResult.country} Scenario</span></p>
                          </div>
                          <div className="text-right">
                            <span className="text-xs font-mono text-slate-400">REF_ID: #492026-HDI</span>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                          {/* Score Card */}
                          <div className="bg-white border border-slate-200 p-8 rounded-2xl flex flex-col items-center justify-center relative overflow-hidden shadow-xs">
                            <div className="absolute top-0 right-0 p-4">
                              <div className="w-3 h-3 rounded-full bg-emerald-500 animate-pulse"></div>
                            </div>
                            <span className="text-xs text-slate-400 uppercase font-bold tracking-widest mb-2">Calculated HDI Score</span>
                            <div className="text-8xl font-black text-slate-900 tracking-tighter font-mono">{predResult.formattedScore}</div>
                            <div className="mt-6 bg-emerald-100 text-emerald-800 px-4 py-1 rounded-full text-xs font-black uppercase tracking-tighter">
                              {predResult.predictedCategory} Development
                            </div>
                          </div>

                          {/* Interpretation Card */}
                          <div className="space-y-6">
                            <div className="bg-slate-900 text-white p-6 rounded-2xl shadow-lg border border-slate-800">
                              <h3 className="text-xs font-bold opacity-50 uppercase tracking-widest mb-3">Insight Analysis</h3>
                              <p className="text-xs leading-relaxed opacity-90">
                                The prediction pipeline evaluated socioeconomic inputs for **{predResult.country}**. A projected HDI score of **{predResult.formattedScore}** places this region in the **{predResult.predictedCategory}** tier globally.
                              </p>
                            </div>

                            <div className="bg-white border border-slate-200 p-6 rounded-2xl shadow-xs">
                              <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">Visual Index</h3>
                              <div className="relative h-12 bg-slate-100 rounded-full overflow-hidden flex items-center px-2 border border-slate-200">
                                <div className="h-2 bg-rose-400 w-[20%] rounded-l-full"></div>
                                <div className="h-2 bg-amber-400 w-[20%]"></div>
                                <div className="h-2 bg-blue-400 w-[20%]"></div>
                                <div className="h-2 bg-emerald-400 w-[40%] rounded-r-full"></div>
                                <div 
                                  className="absolute h-8 w-1 bg-slate-900 shadow-md transform -translate-x-1/2 transition-all duration-500"
                                  style={{ left: `${Math.min(98, Math.max(2, predResult.predictedScore * 100))}%` }}
                                ></div>
                              </div>
                              <div className="flex justify-between mt-3 text-[10px] font-bold text-slate-400 uppercase tracking-tighter">
                                <span>Low</span>
                                <span>Medium</span>
                                <span>High</span>
                                <span>Very High</span>
                              </div>
                            </div>
                          </div>
                        </div>

                        {/* Metrics Table */}
                        <div className="mt-10 bg-white border border-slate-200 rounded-xl p-6 shadow-xs space-y-4">
                          <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest">Socioeconomic Factors Input Summary</h3>
                          <div className="grid grid-cols-2 sm:grid-cols-4 gap-0 border border-slate-200 rounded-lg overflow-hidden font-mono">
                            <div className="bg-white p-4 border-r border-b sm:border-b-0 border-slate-200">
                              <span className="block text-[9px] text-slate-400 uppercase font-bold font-sans mb-1">Life Exp</span>
                              <span className="text-base font-bold text-slate-800">{predResult.lifeExpectancy} Yrs</span>
                            </div>
                            <div className="bg-white p-4 border-r border-b sm:border-b-0 border-slate-200">
                              <span className="block text-[9px] text-slate-400 uppercase font-bold font-sans mb-1">Exp Schooling</span>
                              <span className="text-base font-bold text-slate-800">{predResult.expectedSchooling} Yrs</span>
                            </div>
                            <div className="bg-white p-4 border-r border-slate-200">
                              <span className="block text-[9px] text-slate-400 uppercase font-bold font-sans mb-1">Mean Schooling</span>
                              <span className="text-base font-bold text-slate-800">{predResult.meanSchooling} Yrs</span>
                            </div>
                            <div className="bg-slate-50 p-4">
                              <span className="block text-[9px] text-indigo-400 uppercase font-bold font-sans mb-1">GNI Per Capita</span>
                              <span className="text-base font-bold text-indigo-600">${predResult.gniPerCapita.toLocaleString()}</span>
                            </div>
                          </div>
                        </div>

                        {/* AI-Powered Policy Insights Card */}
                        <div className="bg-white border border-slate-200 rounded-xl p-6 sm:p-8 shadow-xs space-y-4">
                          <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
                            <Brain className="w-5.5 h-5.5 text-indigo-600 shrink-0" />
                            <h4 className="text-sm font-bold text-slate-900 uppercase tracking-wider">AI-Powered Policy Analysis & Insights</h4>
                          </div>
                          <div className="space-y-4 text-slate-600">
                            {renderMarkdown(predResult.insights)}
                          </div>
                        </div>

                      </motion.div>
                    ) : (
                      /* Initial Empty State */
                      <div className="bg-white/50 border border-slate-200/60 rounded-xl p-12 text-center flex flex-col items-center justify-center gap-4 h-full min-h-[350px]">
                        <div className="p-4 bg-white border border-slate-200 rounded-full text-slate-400 shadow-sm">
                          <Brain className="w-8 h-8" />
                        </div>
                        <div className="space-y-1 max-w-sm">
                          <h4 className="text-sm font-bold text-slate-800 uppercase tracking-widest">Awaiting Engine Run</h4>
                          <p className="text-xs text-slate-500 leading-relaxed">
                            Fill out the socioeconomic parameters form on the left or load one of our predefined country template scenarios, then run the calculation engine to generate predictions and compile reports.
                          </p>
                        </div>
                      </div>
                    )}
                  </section>
                </motion.div>
              )}

            </AnimatePresence>
          )}
        </div>

        {/* Footer Info Bar */}
        <footer className="h-10 bg-white border-t border-slate-200 px-8 flex items-center justify-between text-[10px] text-slate-400 font-medium shrink-0">
          <div className="flex gap-6">
            <span>DATASET: UNDP_HDR_2022.csv</span>
            <span>LAST RETRAINED: 03 JUL 2026</span>
          </div>
          <div className="flex gap-2 items-center">
            <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
            <span className="uppercase tracking-tighter">SYSTEM STATUS: OPTIMAL</span>
          </div>
        </footer>
      </main>

    </div>
  );
}
