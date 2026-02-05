import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { predictReview, explainReview, visualizeReview } from '../utils/api'
import ResultCard from './ResultCard'
import { AlertCircle, Sparkles, Zap, TrendingUp, BarChart3 } from 'lucide-react'

export default function ReviewForm({ showToast }) {
  const [review, setReview] = useState('')
  const [result, setResult] = useState(null)
  const [explanation, setExplanation] = useState(null)
  const [visualization, setVisualization] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingStage, setLoadingStage] = useState('')
  const [focused, setFocused] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    setResult(null)
    setExplanation(null)
    setVisualization(null)

    if (!review.trim()) {
      showToast?.('Please enter a review to analyze', 'error')
      return
    }

    if (review.trim().length < 10) {
      showToast?.('Review is too short. Please enter at least 10 characters.', 'error')
      return
    }

    try {
      setLoading(true)
      
      // Stage 1: Prediction
      setLoadingStage('Analyzing review...')
      const predictionData = await predictReview(review)
      setResult(predictionData)
      
      // Stage 2: Explanation (always fetch, but don't fail if it errors)
      try {
        setLoadingStage('Generating explanation...')
        const explanationData = await explainReview(review)
        setExplanation(explanationData)
      } catch (err) {
        console.warn('Explanation failed:', err)
      }

      // Stage 3: Always fetch visualization
      try {
        setLoadingStage('Creating visualization...')
        const vizData = await visualizeReview(review)
        setVisualization(vizData)
      } catch (err) {
        console.warn('Visualization failed:', err)
      }
      
      setLoadingStage('Complete!')
      showToast?.('Analysis complete!', 'success')
    } catch (err) {
      console.error('Analysis error:', err)
      showToast?.(err.message || 'Analysis failed. Please check if the API is running.', 'error')
    } finally {
      setLoading(false)
      setLoadingStage('')
    }
  }

  const clearForm = () => {
    setReview('')
    setResult(null)
    setExplanation(null)
    setVisualization(null)
  }

  const loadExample = () => {
    const examples = [
      "This product exceeded my expectations! The quality is outstanding and it arrived quickly. I've been using it for a month now and it works perfectly. Highly recommend to anyone looking for a reliable product.",
      "AMAZING!!! Best product ever!!! Everyone should buy this NOW!!! 5 stars 5 stars 5 stars!!!",
      "I received this product as a gift and have been thoroughly impressed with its performance. The build quality is solid and it does exactly what it promises. Good value for money.",
      "wow great awesome perfect excellent wonderful fantastic superb incredible outstanding magnificent"
    ]
    setReview(examples[Math.floor(Math.random() * examples.length)])
  }

  const wordCount = review.trim().split(/\s+/).filter(w => w.length > 0).length
  const charCount = review.length

  return (
    <motion.div
      className="w-full"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="text-center mb-8"
      >
        <motion.div
          className="inline-flex items-center gap-2 px-4 py-2 mb-4 bg-purple-500/10 dark:bg-purple-500/20 border border-purple-500/30 rounded-full"
          whileHover={{ scale: 1.05 }}
        >
          <Sparkles className="w-4 h-4 text-purple-600 dark:text-purple-400" />
          <span className="text-sm font-semibold text-purple-700 dark:text-purple-300">
            AI-Powered Analysis
          </span>
          <motion.div
            className="w-2 h-2 bg-purple-500 rounded-full"
            animate={{ opacity: [1, 0.3, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
        </motion.div>

        <h2 className="text-2xl md:text-3xl font-bold text-gray-900 dark:text-gray-100 mb-2">
          Analyze Your Review
        </h2>
        <p className="text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
          Paste any product review below to detect if it's authentic or fake
        </p>
      </motion.div>

      <motion.form
        onSubmit={submit}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        {/* Main Input Area */}
        <motion.div
          className="relative group"
          whileHover={{ scale: showAdvanced ? 1 : 1.005 }}
          transition={{ type: "spring", stiffness: 300 }}
        >
          {/* Glow effect when focused */}
          <motion.div
            className={`absolute -inset-1 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 rounded-2xl blur transition-opacity duration-300 ${
              focused ? 'opacity-30' : 'opacity-0 group-hover:opacity-20'
            }`}
          />

          <div className="relative bg-white dark:bg-gray-900 border-2 border-gray-200 dark:border-gray-800 rounded-2xl p-6 shadow-lg">
            {/* Textarea */}
            <div className="relative mb-4">
              <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
                Review Text
              </label>
              
              <textarea
                value={review}
                onChange={(e) => setReview(e.target.value)}
                onFocus={() => setFocused(true)}
                onBlur={() => setFocused(false)}
                placeholder="Paste your review here... (minimum 10 characters)"
                rows={6}
                className="w-full resize-none rounded-xl border-2 border-gray-300 dark:border-gray-700 
                         bg-gray-50 dark:bg-gray-800 px-4 py-3 text-gray-900 dark:text-gray-100
                         placeholder-gray-400 dark:placeholder-gray-500 outline-none 
                         focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500 
                         transition-all duration-200"
              />
              
              {/* Stats */}
              <AnimatePresence>
                {review.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: -5 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -5 }}
                    className="flex items-center justify-between mt-2 text-xs text-gray-500 dark:text-gray-400"
                  >
                    <div className="flex items-center gap-4">
                      <span>{charCount} characters</span>
                      <span>•</span>
                      <span>{wordCount} words</span>
                    </div>
                    {charCount < 10 && (
                      <span className="text-orange-500 dark:text-orange-400 flex items-center gap-1">
                        <AlertCircle size={12} />
                        Too short
                      </span>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Advanced Options Toggle */}
            <motion.button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 text-sm font-medium text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 mb-4"
              whileHover={{ x: 3 }}
            >
              <BarChart3 size={16} />
              {showAdvanced ? 'Hide' : 'Show'} Advanced Features
              <motion.span
                animate={{ rotate: showAdvanced ? 180 : 0 }}
                transition={{ duration: 0.3 }}
              >
                ▼
              </motion.span>
            </motion.button>

            {/* Advanced Options */}
            <AnimatePresence>
              {showAdvanced && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.3 }}
                  className="mb-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700"
                >
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                    <strong>Enabled:</strong> SHAP Explanations + Confidence Visualization
                  </p>
                  <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-500">
                    <TrendingUp size={14} />
                    <span>Shows which words influenced the prediction</span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Action Buttons */}
            <div className="flex flex-wrap items-center gap-3">
              <motion.button
                whileTap={{ scale: 0.98 }}
                whileHover={{ scale: 1.02 }}
                type="submit"
                disabled={loading || !review.trim() || charCount < 10}
                className="btn-primary flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <>
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                    />
                    <span>{loadingStage || 'Analyzing...'}</span>
                  </>
                ) : (
                  <>
                    <Zap size={20} />
                    <span>Analyze Review</span>
                  </>
                )}
              </motion.button>

              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                type="button"
                onClick={loadExample}
                className="btn-secondary flex items-center gap-2"
              >
                <Sparkles size={18} />
                <span className="hidden sm:inline">Load Example</span>
                <span className="sm:hidden">Example</span>
              </motion.button>

              {review && (
                <motion.button
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  type="button"
                  onClick={clearForm}
                  className="px-4 py-2 rounded-xl text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 font-medium transition-colors"
                >
                  Clear
                </motion.button>
              )}
            </div>

            {/* Loading Progress */}
            {loading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-4"
              >
                <div className="h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600"
                    initial={{ width: "0%" }}
                    animate={{ width: "100%" }}
                    transition={{ duration: 2, ease: "easeInOut" }}
                  />
                </div>
              </motion.div>
            )}
          </div>
        </motion.div>
      </motion.form>

      {/* Results Display */}
      <AnimatePresence mode="wait">
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            transition={{ duration: 0.5 }}
            className="mt-8"
          >
            <ResultCard 
              data={result} 
              explanation={explanation}
              visualization={visualization}
              review={review}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}