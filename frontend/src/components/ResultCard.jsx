import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { CheckCircle2, XCircle, Shield, AlertTriangle, TrendingUp, Brain, Eye, ChevronDown, ChevronUp, BarChart3, Sparkles, PieChart, Activity, Target, Gauge } from "lucide-react"

// Probability Bar Component
function ProbabilityBar({ label, value, color }) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center text-sm">
        <span className="font-semibold text-gray-700 dark:text-gray-300">{label}</span>
        <span className={`font-bold ${color === 'green' ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 1, ease: "easeOut" }}
          className={`h-full rounded-full ${color === 'green' ? 'bg-gradient-to-r from-green-500 to-emerald-500' : 'bg-gradient-to-r from-red-500 to-pink-500'}`}
        />
      </div>
    </div>
  )
}

// SENTIMENT ANALYSIS GAUGE
function SentimentGauge({ review }) {
  const calculateSentiment = () => {
    if (!review) return 0
    
    const positiveWords = ['great', 'excellent', 'amazing', 'perfect', 'love', 'best', 'wonderful', 'fantastic', 'awesome', 'good']
    const negativeWords = ['bad', 'terrible', 'worst', 'hate', 'poor', 'awful', 'horrible', 'disappointing', 'waste']
    
    const words = review.toLowerCase().split(/\s+/)
    const positiveCount = words.filter(w => positiveWords.some(pw => w.includes(pw))).length
    const negativeCount = words.filter(w => negativeWords.some(nw => w.includes(nw))).length
    const exclamationCount = (review.match(/!/g) || []).length
    
    const totalSentiment = positiveCount - negativeCount
    const sentimentScore = Math.max(-1, Math.min(1, totalSentiment / Math.max(words.length / 10, 1)))
    const punctuationBoost = exclamationCount > 3 ? 0.3 : 0
    
    return Math.max(-1, Math.min(1, sentimentScore + punctuationBoost))
  }

  const sentiment = calculateSentiment()
  const sentimentPercent = ((sentiment + 1) / 2) * 100
  const rotation = -90 + (sentimentPercent / 100) * 180

  const getSentimentColor = () => {
    if (sentiment < -0.3) return { text: 'text-red-600 dark:text-red-400', label: 'Negative' }
    if (sentiment < 0.3) return { text: 'text-yellow-600 dark:text-yellow-400', label: 'Neutral' }
    if (sentiment < 0.7) return { text: 'text-green-600 dark:text-green-400', label: 'Positive' }
    return { text: 'text-purple-600 dark:text-purple-400', label: 'Overly Positive' }
  }

  const color = getSentimentColor()

  return (
    <div className="space-y-4">
      <div className="flex flex-col items-center">
        <div className="relative w-48 h-24">
          <svg viewBox="0 0 200 100" className="w-full h-full">
            <path d="M 20 90 A 80 80 0 0 1 60 25" fill="none" stroke="currentColor" strokeWidth="12" className="text-red-500/30" />
            <path d="M 60 25 A 80 80 0 0 1 100 10" fill="none" stroke="currentColor" strokeWidth="12" className="text-yellow-500/30" />
            <path d="M 100 10 A 80 80 0 0 1 140 25" fill="none" stroke="currentColor" strokeWidth="12" className="text-green-500/30" />
            <path d="M 140 25 A 80 80 0 0 1 180 90" fill="none" stroke="currentColor" strokeWidth="12" className="text-purple-500/30" />
            
            <motion.g
              initial={{ rotate: -90 }}
              animate={{ rotate: rotation }}
              transition={{ duration: 1.5, type: "spring", stiffness: 60 }}
              style={{ transformOrigin: "100px 90px" }}
            >
              <line x1="100" y1="90" x2="100" y2="30" stroke="currentColor" strokeWidth="3" strokeLinecap="round" className="text-gray-800 dark:text-gray-200" />
              <circle cx="100" cy="90" r="6" fill="currentColor" className="text-gray-800 dark:text-gray-200" />
            </motion.g>
          </svg>
        </div>
        
        <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.5 }} className={`text-2xl font-black ${color.text} mt-2`}>
          {sentiment.toFixed(2)}
        </motion.div>
        <div className="text-xs text-gray-500 dark:text-gray-400 font-semibold">{color.label}</div>
        
        {sentiment > 0.7 && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-3 p-2 bg-purple-500/10 border border-purple-500/30 rounded-lg">
            <p className="text-xs text-purple-700 dark:text-purple-300 flex items-center gap-1">
              <AlertTriangle size={12} />
              May indicate fake
            </p>
          </motion.div>
        )}
      </div>
    </div>
  )
}

// TEXT COMPLEXITY RADAR
function TextComplexityRadar({ review, reviewStats }) {
  const calculateComplexityMetrics = () => {
    if (!review || !reviewStats) return null

    const words = review.split(/\s+/).filter(w => w.length > 0)
    const sentences = review.split(/[.!?]+/).filter(s => s.trim().length > 0)
    
    const uniqueWords = new Set(words.map(w => w.toLowerCase()))
    const vocabularyDiversity = Math.min(100, (uniqueWords.size / words.length) * 100)
    const avgWordsPerSentence = words.length / Math.max(sentences.length, 1)
    const sentenceComplexity = Math.min(100, (avgWordsPerSentence / 20) * 100)
    const avgWordLength = reviewStats.avg_word_length || 5
    const readability = Math.min(100, 100 - ((avgWordLength - 4) / 6) * 100)
    const punctuationCount = (review.match(/[.,!?;:]/g) || []).length
    const punctuationScore = Math.min(100, (punctuationCount / words.length) * 100 * 10)
    const properCapitalization = review.match(/[A-Z][a-z]+/g) || []
    const capitalizationScore = Math.min(100, (properCapitalization.length / words.length) * 100 * 2)
    const wordFreq = {}
    words.forEach(w => { wordFreq[w.toLowerCase()] = (wordFreq[w.toLowerCase()] || 0) + 1 })
    const maxFreq = Math.max(...Object.values(wordFreq))
    const repetitionScore = Math.max(0, 100 - (maxFreq / words.length) * 100 * 5)

    return { vocabularyDiversity, sentenceComplexity, readability, punctuation: punctuationScore, capitalization: capitalizationScore, repetition: repetitionScore }
  }

  const metrics = calculateComplexityMetrics()
  if (!metrics) return null

  const dimensions = [
    { label: 'Vocab', key: 'vocabularyDiversity', angle: 0 },
    { label: 'Sentence', key: 'sentenceComplexity', angle: 60 },
    { label: 'Read', key: 'readability', angle: 120 },
    { label: 'Punct', key: 'punctuation', angle: 180 },
    { label: 'Caps', key: 'capitalization', angle: 240 },
    { label: 'Unique', key: 'repetition', angle: 300 }
  ]

  const center = 100
  const maxRadius = 70
  const scale = maxRadius / 100

  const getPoint = (value, angle) => {
    const radian = (angle - 90) * (Math.PI / 180)
    const radius = (value || 0) * scale
    return { x: center + radius * Math.cos(radian), y: center + radius * Math.sin(radian) }
  }

  const points = dimensions.map(d => getPoint(metrics[d.key], d.angle))
  const pathData = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ') + ' Z'

  return (
    <div className="space-y-4">
      <div className="flex flex-col items-center">
        <svg viewBox="0 0 200 200" className="w-full max-w-xs">
          {[20, 40, 60].map((r, i) => (
            <circle key={i} cx={center} cy={center} r={r} fill="none" stroke="currentColor" strokeWidth="1" className="text-gray-300 dark:text-gray-700" opacity={0.3} />
          ))}
          
          {dimensions.map((d, i) => {
            const endPoint = getPoint(100, d.angle)
            return <line key={i} x1={center} y1={center} x2={endPoint.x} y2={endPoint.y} stroke="currentColor" strokeWidth="1" className="text-gray-300 dark:text-gray-700" opacity={0.5} />
          })}
          
          <motion.path
            d={pathData}
            fill="currentColor"
            className="text-blue-500 dark:text-blue-400"
            opacity={0.3}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ duration: 1 }}
            style={{ transformOrigin: `${center}px ${center}px` }}
          />
          <motion.path d={pathData} fill="none" stroke="currentColor" strokeWidth="2" className="text-blue-500" initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.5 }} />
          
          {points.map((p, i) => (
            <motion.circle key={i} cx={p.x} cy={p.y} r="3" fill="currentColor" className="text-blue-600" initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.2 + i * 0.1 }} />
          ))}
          
          {dimensions.map((d, i) => {
            const labelPoint = getPoint(85, d.angle)
            return <text key={i} x={labelPoint.x} y={labelPoint.y} textAnchor="middle" className="text-[10px] font-semibold fill-gray-700 dark:fill-gray-300">{d.label}</text>
          })}
        </svg>
      </div>
    </div>
  )
}

// RISK ASSESSMENT
function RiskAssessmentGauge({ data, review, reviewStats }) {
  const calculateRisk = () => {
    let totalRisk = 0
    const factors = []

    if (data.confidence < 0.7) {
      const risk = (1 - data.confidence) * 40
      factors.push({ name: 'Low Confidence', score: risk, icon: '⚠️' })
      totalRisk += risk
    }
    
    if (reviewStats) {
      const exclamationRatio = reviewStats.exclamation_count / reviewStats.word_count
      if (exclamationRatio > 0.05) {
        const risk = Math.min(25, exclamationRatio * 100)
        factors.push({ name: 'Excessive !', score: risk, icon: '❗' })
        totalRisk += risk
      }
      
      if (reviewStats.uppercase_ratio > 0.15) {
        const risk = Math.min(20, reviewStats.uppercase_ratio * 50)
        factors.push({ name: 'Too Many CAPS', score: risk, icon: '🔤' })
        totalRisk += risk
      }
      
      if (reviewStats.word_count < 10) {
        factors.push({ name: 'Very Short', score: 15, icon: '📝' })
        totalRisk += 15
      }
    }
    
    if (data.prediction === 0) {
      const risk = data.probability.fake * 30
      factors.push({ name: 'Classified Fake', score: risk, icon: '❌' })
      totalRisk += risk
    }

    totalRisk = Math.min(100, totalRisk)
    const level = totalRisk < 20 ? 'Low' : totalRisk < 50 ? 'Medium' : totalRisk < 75 ? 'High' : 'Critical'
    
    return { factors, totalRisk, riskLevel: level }
  }

  const risk = calculateRisk()
  const rotation = -90 + (risk.totalRisk / 100) * 180

  const getColor = () => {
    if (risk.riskLevel === 'Low') return { text: 'text-green-600 dark:text-green-400', bg: 'bg-green-500' }
    if (risk.riskLevel === 'Medium') return { text: 'text-yellow-600 dark:text-yellow-400', bg: 'bg-yellow-500' }
    if (risk.riskLevel === 'High') return { text: 'text-orange-600 dark:text-orange-400', bg: 'bg-orange-500' }
    return { text: 'text-red-600 dark:text-red-400', bg: 'bg-red-500' }
  }

  const color = getColor()

  return (
    <div className="space-y-4">
      <div className="flex flex-col items-center">
        <div className="relative w-56 h-28">
          <svg viewBox="0 0 240 120" className="w-full h-full">
            <path d="M 30 110 A 90 90 0 0 1 75 35" fill="none" strokeWidth="16" className="stroke-green-500/30" />
            <path d="M 75 35 A 90 90 0 0 1 120 15" fill="none" strokeWidth="16" className="stroke-yellow-500/30" />
            <path d="M 120 15 A 90 90 0 0 1 165 35" fill="none" strokeWidth="16" className="stroke-orange-500/30" />
            <path d="M 165 35 A 90 90 0 0 1 210 110" fill="none" strokeWidth="16" className="stroke-red-500/30" />
            
            <motion.g initial={{ rotate: -90 }} animate={{ rotate: rotation }} transition={{ duration: 2, type: "spring" }} style={{ transformOrigin: "120px 110px" }}>
              <path d="M 120 110 L 115 50 L 120 45 L 125 50 Z" fill="currentColor" className="text-gray-800 dark:text-gray-200" />
              <circle cx="120" cy="110" r="6" fill="currentColor" className="text-gray-800 dark:text-gray-200" />
            </motion.g>
          </svg>
        </div>
        
        <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.5 }} className="text-center mt-2">
          <div className={`text-3xl font-black ${color.text}`}>{risk.totalRisk.toFixed(0)}%</div>
          <div className={`text-sm font-bold ${color.text}`}>{risk.riskLevel} Risk</div>
        </motion.div>
        
        {risk.factors.length > 0 && (
          <div className="mt-4 space-y-2 w-full">
            {risk.factors.slice(0, 3).map((f, i) => (
              <div key={i} className="flex items-center justify-between text-xs p-2 bg-gray-100 dark:bg-gray-800 rounded-lg">
                <span className="flex items-center gap-1">
                  <span>{f.icon}</span>
                  <span className="font-medium text-gray-700 dark:text-gray-300">{f.name}</span>
                </span>
                <span className="font-bold text-red-600 dark:text-red-400">+{f.score.toFixed(0)}%</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// Model Comparison Chart Component
function ModelComparisonChart({ modelPredictions }) {
  if (!modelPredictions) return null

  const models = Object.entries(modelPredictions).map(([name, data]) => ({
    name: name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    confidence: data.confidence || 0.5,
    prediction: data.prediction,
    probReal: data.probabilities?.real || (data.prediction === 1 ? data.confidence : 1 - data.confidence)
  }))

  return (
    <div className="space-y-4">
      <h4 className="text-lg font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2">
        <BarChart3 size={20} className="text-purple-600 dark:text-purple-400" />
        Model Comparison
      </h4>
      
      <div className="grid gap-3">
        {models.map((model, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: idx * 0.1 }}
            className="relative"
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${model.prediction === 1 ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="font-semibold text-sm text-gray-700 dark:text-gray-300">
                  {model.name}
                </span>
              </div>
              <span className="text-xs font-bold text-gray-600 dark:text-gray-400">
                {(model.confidence * 100).toFixed(1)}%
              </span>
            </div>
            
            <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded-lg overflow-hidden">
              <div className="flex h-full">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(1 - model.probReal) * 100}%` }}
                  transition={{ duration: 0.8, delay: idx * 0.1 }}
                  className="bg-gradient-to-r from-red-500 to-pink-500 flex items-center justify-center"
                >
                  {(1 - model.probReal) > 0.15 && (
                    <span className="text-xs font-bold text-white">
                      Fake {((1 - model.probReal) * 100).toFixed(0)}%
                    </span>
                  )}
                </motion.div>
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${model.probReal * 100}%` }}
                  transition={{ duration: 0.8, delay: idx * 0.1 }}
                  className="bg-gradient-to-r from-green-500 to-emerald-500 flex items-center justify-center"
                >
                  {model.probReal > 0.15 && (
                    <span className="text-xs font-bold text-white">
                      Real {(model.probReal * 100).toFixed(0)}%
                    </span>
                  )}
                </motion.div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}

// Word Importance Chart Component
function WordImportanceChart({ words }) {
  if (!words || words.length === 0) return null

  const maxWeight = Math.max(...words.map(w => Math.abs(w.weight)))

  return (
    <div className="space-y-4">
      <h4 className="text-lg font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2">
        <Activity size={20} className="text-blue-600 dark:text-blue-400" />
        Word Importance
      </h4>
      
      <div className="space-y-2">
        {words.slice(0, 10).map((word, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.05 }}
            className="flex items-center gap-3"
          >
            <div className="w-24 text-right">
              <span className="text-sm font-semibold text-gray-700 dark:text-gray-300 truncate block">
                {word.word}
              </span>
            </div>
            
            <div className="flex-1 h-6 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${(Math.abs(word.weight) / maxWeight) * 100}%` }}
                transition={{ duration: 0.8, delay: idx * 0.05 }}
                className={`h-full flex items-center px-2 ${
                  word.impact === 'fake' 
                    ? 'bg-gradient-to-r from-red-500 to-pink-500' 
                    : 'bg-gradient-to-r from-green-500 to-emerald-500'
                }`}
              >
                <span className="text-xs font-bold text-white whitespace-nowrap">
                  {word.weight > 0 ? '+' : ''}{(word.weight * 100).toFixed(0)}%
                </span>
              </motion.div>
            </div>
            
            <div className={`w-16 text-xs font-bold text-right ${
              word.impact === 'fake' ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'
            }`}>
              {word.impact}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}

// Review Stats Visualization Component
function ReviewStatsChart({ stats }) {
  if (!stats) return null

  const metrics = [
    { label: 'Word Count', value: stats.word_count, max: 200, color: 'from-blue-500 to-cyan-500', icon: '📝' },
    { label: 'Avg Word Length', value: stats.avg_word_length || 0, max: 10, color: 'from-purple-500 to-pink-500', icon: '📏' },
    { label: 'Uppercase Ratio', value: (stats.uppercase_ratio * 100), max: 100, color: 'from-orange-500 to-red-500', icon: '🔤' },
    { label: 'Exclamation Count', value: stats.exclamation_count, max: 10, color: 'from-yellow-500 to-orange-500', icon: '❗' }
  ]

  return (
    <div className="space-y-4">
      <h4 className="text-lg font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2">
        <PieChart size={20} className="text-green-600 dark:text-green-400" />
        Review Statistics
      </h4>
      
      <div className="grid grid-cols-2 gap-4">
        {metrics.map((metric, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: idx * 0.1 }}
            className="p-4 bg-gray-50 dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-2xl">{metric.icon}</span>
              <span className="text-xl font-black text-gray-900 dark:text-gray-100">
                {typeof metric.value === 'number' ? metric.value.toFixed(metric.max > 50 ? 0 : 1) : metric.value}
              </span>
            </div>
            <div className="text-xs font-semibold text-gray-600 dark:text-gray-400 mb-2">
              {metric.label}
            </div>
            <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${Math.min((metric.value / metric.max) * 100, 100)}%` }}
                transition={{ duration: 1, delay: idx * 0.1 }}
                className={`h-full bg-gradient-to-r ${metric.color}`}
              />
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}

export default function ResultCard({ data, explanation, visualization, review }) {
  const [showExplanation, setShowExplanation] = useState(false)
  const [selectedWord, setSelectedWord] = useState(null)
  const [showVisualization, setShowVisualization] = useState(true)
  const [showModelComparison, setShowModelComparison] = useState(true)
  const [showWordChart, setShowWordChart] = useState(true)
  const [showStatsChart, setShowStatsChart] = useState(true)

  if (!data) return null

  const isReal = data.prediction === 1
  const confidence = data.confidence || Math.max(data.probability.real, data.probability.fake)
  const confidenceLevel = data.confidence_level || (confidence >= 0.85 ? "High" : confidence >= 0.70 ? "Medium" : "Low")
  const hasMultipleModels = data.model_predictions && Object.keys(data.model_predictions).length > 1

  return (
    <div className="w-full space-y-6">
      {/* Main Result Card */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="relative"
      >
        <div className={`absolute -inset-1 rounded-3xl blur-lg opacity-30 ${
          isReal 
            ? "bg-gradient-to-r from-green-500 to-emerald-500"
            : "bg-gradient-to-r from-red-500 to-pink-500"
        }`}></div>
        
        <div className="relative glass-card p-8 shadow-2xl">
          {/* Header */}
          <div className="flex items-center justify-between mb-8">
            <div className="flex items-center gap-4">
              <motion.div 
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
                className={`w-16 h-16 rounded-2xl flex items-center justify-center shadow-lg ${
                  isReal 
                    ? "bg-gradient-to-br from-green-500 to-emerald-600"
                    : "bg-gradient-to-br from-red-500 to-pink-600"
                }`}
              >
                {isReal ? (
                  <CheckCircle2 size={32} className="text-white" strokeWidth={2.5} />
                ) : (
                  <XCircle size={32} className="text-white" strokeWidth={2.5} />
                )}
              </motion.div>
              
              <div>
                <motion.h3 
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 }}
                  className="text-3xl font-black text-gray-900 dark:text-gray-100"
                >
                  {data.result}
                </motion.h3>
                <motion.p 
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.4 }}
                  className="text-gray-600 dark:text-gray-400 font-medium"
                >
                  Analysis Complete
                </motion.p>
              </div>
            </div>
            
            <motion.div
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.5, type: "spring" }}
              className={`px-4 py-2 rounded-xl font-bold text-sm border ${
                confidenceLevel === "High" 
                  ? "bg-green-500/10 dark:bg-green-500/20 border-green-500/30 text-green-700 dark:text-green-400"
                  : confidenceLevel === "Medium"
                  ? "bg-yellow-500/10 dark:bg-yellow-500/20 border-yellow-500/30 text-yellow-700 dark:text-yellow-400"
                  : "bg-orange-500/10 dark:bg-orange-500/20 border-orange-500/30 text-orange-700 dark:text-orange-400"
              }`}
            >
              {confidenceLevel} Confidence
            </motion.div>
          </div>

          {/* Confidence Circle */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="flex justify-center mb-8"
          >
            <div className="relative w-48 h-48">
              <svg className="w-full h-full transform -rotate-90">
                <circle
                  cx="96"
                  cy="96"
                  r="88"
                  stroke="currentColor"
                  strokeWidth="12"
                  className="text-gray-200 dark:text-gray-800"
                  fill="transparent"
                />
                <motion.circle
                  cx="96"
                  cy="96"
                  r="88"
                  stroke="currentColor"
                  strokeWidth="12"
                  strokeLinecap="round"
                  className={isReal ? "text-green-500" : "text-red-500"}
                  fill="transparent"
                  initial={{ strokeDashoffset: 2 * Math.PI * 88 }}
                  animate={{ strokeDashoffset: 2 * Math.PI * 88 * (1 - confidence) }}
                  transition={{ duration: 1.5, ease: "easeOut" }}
                  style={{ strokeDasharray: 2 * Math.PI * 88 }}
                />
              </svg>
              
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.8, type: "spring", stiffness: 200 }}
                  className={`text-5xl font-black ${
                    isReal ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"
                  }`}
                >
                  {Math.round(confidence * 100)}%
                </motion.div>
                <span className="text-xs text-gray-500 dark:text-gray-400 font-semibold mt-1">
                  CONFIDENCE
                </span>
              </div>
            </div>
          </motion.div>

          {/* Probability Bars */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9 }}
            className="space-y-4 mb-8"
          >
            <ProbabilityBar label="Real" value={data.probability.real} color="green" />
            <ProbabilityBar label="Fake" value={data.probability.fake} color="red" />
          </motion.div>

          {hasMultipleModels && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.1 }}
              className="flex items-center justify-center gap-3 p-4 bg-blue-500/10 dark:bg-blue-500/20 border border-blue-500/30 rounded-xl"
            >
              <Shield size={20} className="text-blue-600 dark:text-blue-400" />
              <span className="text-sm font-semibold text-blue-700 dark:text-blue-300">
                Model Agreement: {data.model_agreement}
              </span>
            </motion.div>
          )}

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.2 }}
            className="flex items-start gap-3 p-4 bg-purple-500/10 dark:bg-purple-500/20 border border-purple-500/30 rounded-xl mt-6"
          >
            <AlertTriangle size={20} className="text-purple-600 dark:text-purple-400 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-gray-700 dark:text-gray-300">
              <strong className="text-purple-700 dark:text-purple-400">Note:</strong> This AI model analyzes linguistic patterns, writing style, and review characteristics. 
              {confidenceLevel === "Low" && " Low confidence suggests the review has mixed signals and should be evaluated carefully."}
            </div>
          </motion.div>
        </div>
      </motion.div>

      {/* NEW: Advanced Analytics Section - 3 Gauges */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        className="relative"
      >
        <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-3xl blur-lg opacity-20"></div>
        <div className="relative glass-card p-8 shadow-2xl">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-12 h-12 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
              <Gauge size={24} className="text-white" strokeWidth={2.5} />
            </div>
            <div>
              <h3 className="text-2xl font-black text-gray-900 dark:text-gray-100">
                Advanced Analytics
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 font-medium">
                Sentiment, Complexity & Risk Assessment
              </p>
            </div>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            {/* Sentiment Gauge */}
            <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-2xl border border-purple-200 dark:border-purple-800">
              <div className="flex items-center gap-2 mb-4">
                <Activity size={18} className="text-purple-600 dark:text-purple-400" />
                <h4 className="text-md font-bold text-gray-900 dark:text-gray-100">Sentiment</h4>
              </div>
              <SentimentGauge review={review} />
            </div>

            {/* Complexity Radar */}
            <div className="p-4 bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-2xl border border-blue-200 dark:border-blue-800">
              <div className="flex items-center gap-2 mb-4">
                <Target size={18} className="text-blue-600 dark:text-blue-400" />
                <h4 className="text-md font-bold text-gray-900 dark:text-gray-100">Complexity</h4>
              </div>
              <TextComplexityRadar review={review} reviewStats={data.review_stats} />
            </div>

            {/* Risk Assessment */}
            <div className="p-4 bg-gradient-to-br from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 rounded-2xl border border-red-200 dark:border-red-800">
              <div className="flex items-center gap-2 mb-4">
                <Shield size={18} className="text-red-600 dark:text-red-400" />
                <h4 className="text-md font-bold text-gray-900 dark:text-gray-100">Risk Level</h4>
              </div>
              <RiskAssessmentGauge data={data} review={review} reviewStats={data.review_stats} />
            </div>
          </div>
        </div>
      </motion.div>

      {/* Review Statistics Chart */}
      {data.review_stats && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="relative"
        >
          <div className="absolute -inset-1 bg-gradient-to-r from-green-500 to-teal-500 rounded-3xl blur-lg opacity-20"></div>
          <div className="relative glass-card p-8 shadow-2xl">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-teal-600 rounded-xl flex items-center justify-center shadow-lg">
                  <PieChart size={24} className="text-white" strokeWidth={2.5} />
                </div>
                <div>
                  <h3 className="text-2xl font-black text-gray-900 dark:text-gray-100">
                    Review Analysis
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 font-medium">
                    Statistical Breakdown
                  </p>
                </div>
              </div>
              <button
                onClick={() => setShowStatsChart(!showStatsChart)}
                className="flex items-center gap-2 px-4 py-2 bg-green-500/10 dark:bg-green-500/20 border border-green-500/30 rounded-xl text-green-700 dark:text-green-300 font-semibold text-sm hover:bg-green-500/20 dark:hover:bg-green-500/30 transition-all"
              >
                {showStatsChart ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>
            </div>

            <AnimatePresence>
              {showStatsChart && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                >
                  <ReviewStatsChart stats={data.review_stats} />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      )}

      {/* Model Comparison Chart */}
      {hasMultipleModels && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          className="relative"
        >
          <div className="absolute -inset-1 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-3xl blur-lg opacity-20"></div>
          <div className="relative glass-card p-8 shadow-2xl">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-xl flex items-center justify-center shadow-lg">
                  <BarChart3 size={24} className="text-white" strokeWidth={2.5} />
                </div>
                <div>
                  <h3 className="text-2xl font-black text-gray-900 dark:text-gray-100">
                    Multi-Model Analysis
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 font-medium">
                    {Object.keys(data.model_predictions).length} Models Compared
                  </p>
                </div>
              </div>
              <button
                onClick={() => setShowModelComparison(!showModelComparison)}
                className="flex items-center gap-2 px-4 py-2 bg-blue-500/10 dark:bg-blue-500/20 border border-blue-500/30 rounded-xl text-blue-700 dark:text-blue-300 font-semibold text-sm hover:bg-blue-500/20 dark:hover:bg-blue-500/30 transition-all"
              >
                {showModelComparison ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>
            </div>

            <AnimatePresence>
              {showModelComparison && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                >
                  <ModelComparisonChart modelPredictions={data.model_predictions} />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      )}

      {/* Enhanced Explanation Card with Word Chart */}
      {explanation && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="relative"
        >
          <div className="absolute -inset-1 bg-gradient-to-r from-purple-600 via-blue-600 to-cyan-600 rounded-3xl blur-lg opacity-20"></div>
          
          <div className="relative glass-card p-8 shadow-2xl">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-gradient-to-br from-purple-600 to-blue-600 rounded-xl flex items-center justify-center shadow-lg">
                  <Brain size={24} className="text-white" strokeWidth={2.5} />
                </div>
                <div>
                  <h3 className="text-2xl font-black text-gray-900 dark:text-gray-100">
                    AI Explanation
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 font-medium">
                    {explanation.explanation_method || "Word Importance Analysis"}
                  </p>
                </div>
              </div>

              <button
                onClick={() => setShowWordChart(!showWordChart)}
                className="flex items-center gap-2 px-4 py-2 bg-purple-500/10 dark:bg-purple-500/20 border border-purple-500/30 rounded-xl text-purple-700 dark:text-purple-300 font-semibold text-sm hover:bg-purple-500/20 dark:hover:bg-purple-500/30 transition-all"
              >
                {showWordChart ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>
            </div>

            {/* Word Importance Chart */}
            <AnimatePresence>
              {showWordChart && explanation.top_words && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mb-6"
                >
                  <WordImportanceChart words={explanation.top_words} />
                </motion.div>
              )}
            </AnimatePresence>

            {/* Top Words Preview */}
            <div className="mb-6">
              <div className="flex items-center gap-2 mb-4">
                <Sparkles size={16} className="text-purple-600 dark:text-purple-400" />
                <span className="text-sm font-bold text-gray-700 dark:text-gray-300">
                  Most Influential Words
                </span>
              </div>
              
              <div className="flex flex-wrap gap-2">
                {explanation.top_words?.slice(0, 8).map((word) => (
                  <button
                    key={word.word}
                    onClick={() => setSelectedWord(selectedWord?.word === word.word ? null : word)}
                    className={`px-3 py-2 rounded-lg border font-medium text-sm transition-all ${
                      selectedWord?.word === word.word
                        ? 'bg-purple-500/20 border-purple-500 text-purple-700 dark:text-purple-300 scale-105'
                        : word.impact === 'fake'
                        ? 'bg-red-500/10 dark:bg-red-500/20 border-red-500/30 text-red-700 dark:text-red-400 hover:bg-red-500/20 dark:hover:bg-red-500/30'
                        : 'bg-green-500/10 dark:bg-green-500/20 border-green-500/30 text-green-700 dark:text-green-400 hover:bg-green-500/20 dark:hover:bg-green-500/30'
                    }`}
                  >
                    {word.word}
                    <span className="ml-1.5 text-xs opacity-75">
                      {word.weight > 0 ? '+' : ''}{(word.weight * 100).toFixed(0)}%
                    </span>
                  </button>
                ))}
              </div>

              <AnimatePresence>
                {selectedWord && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-4 p-4 bg-gray-100 dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-lg font-bold text-gray-900 dark:text-gray-100">
                        "{selectedWord.word}"
                      </span>
                      <span className={`px-3 py-1 rounded-lg text-xs font-bold ${
                        selectedWord.impact === 'fake'
                          ? 'bg-red-500/20 text-red-700 dark:text-red-400'
                          : 'bg-green-500/20 text-green-700 dark:text-green-400'
                      }`}>
                        {selectedWord.impact.toUpperCase()} SIGNAL
                      </span>
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      Influence: <strong className={selectedWord.weight > 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}>
                        {(selectedWord.weight * 100).toFixed(1)}%
                      </strong> towards {selectedWord.impact} classification
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Detailed Explanation */}
            <div className="flex justify-center">
              <button
                onClick={() => setShowExplanation(!showExplanation)}
                className="flex items-center gap-2 px-6 py-3 bg-purple-500/10 dark:bg-purple-500/20 border border-purple-500/30 rounded-xl text-purple-700 dark:text-purple-300 font-semibold hover:bg-purple-500/20 dark:hover:bg-purple-500/30 transition-all"
              >
                <Eye size={18} />
                {showExplanation ? 'Hide' : 'Show'} Full Details
                {showExplanation ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
              </button>
            </div>

            <AnimatePresence>
              {showExplanation && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="border-t border-gray-200 dark:border-gray-800 pt-6 mt-6"
                >
                  <h4 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-4">
                    Complete Word Analysis
                  </h4>
                  
                  <div className="space-y-3">
                    {explanation.top_words?.map((word, index) => (
                      <div
                        key={word.word}
                        className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                      >
                        <div className="flex items-center gap-3">
                          <div className={`w-8 h-8 rounded-lg flex items-center justify-center font-bold text-sm ${
                            word.impact === 'fake'
                              ? 'bg-red-500/20 text-red-700 dark:text-red-400'
                              : 'bg-green-500/20 text-green-700 dark:text-green-400'
                          }`}>
                            {index + 1}
                          </div>
                          <span className="font-semibold text-gray-900 dark:text-gray-100">
                            {word.word}
                          </span>
                        </div>
                        
                        <div className="flex items-center gap-4">
                          <div className="w-32 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-1000 ${
                                word.impact === 'fake' ? 'bg-red-500' : 'bg-green-500'
                              }`}
                              style={{ width: `${Math.abs(word.weight) * 100}%` }}
                            />
                          </div>
                          <span className={`text-sm font-bold w-16 text-right ${
                            word.impact === 'fake'
                              ? 'text-red-600 dark:text-red-400'
                              : 'text-green-600 dark:text-green-400'
                          }`}>
                            {(word.weight * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      )}

      {/* Backend Visualization (Matplotlib Chart) */}
      {visualization && visualization.image && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="relative"
        >
          <div className="absolute -inset-1 bg-gradient-to-r from-orange-500 to-pink-500 rounded-3xl blur-lg opacity-20"></div>
          
          <div className="relative glass-card p-8 shadow-2xl">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-gradient-to-br from-orange-500 to-pink-600 rounded-xl flex items-center justify-center shadow-lg">
                  <BarChart3 size={24} className="text-white" strokeWidth={2.5} />
                </div>
                <div>
                  <h3 className="text-2xl font-black text-gray-900 dark:text-gray-100">
                    Backend Visualization
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 font-medium">
                    Matplotlib Chart Analysis
                  </p>
                </div>
              </div>

              <button
                onClick={() => setShowVisualization(!showVisualization)}
                className="flex items-center gap-2 px-4 py-2 bg-orange-500/10 dark:bg-orange-500/20 border border-orange-500/30 rounded-xl text-orange-700 dark:text-orange-300 font-semibold text-sm hover:bg-orange-500/20 dark:hover:bg-orange-500/30 transition-all"
              >
                {showVisualization ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>
            </div>

            <AnimatePresence>
              {showVisualization && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="rounded-xl overflow-hidden bg-white dark:bg-gray-800 p-4"
                >
                  <img 
                    src={visualization.image} 
                    alt="Prediction visualization from backend" 
                    className="w-full h-auto rounded-lg"
                  />
                  <div className="mt-4 text-center text-sm text-gray-600 dark:text-gray-400">
                    Generated by Flask backend using Matplotlib
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      )}
    </div>
  )
}