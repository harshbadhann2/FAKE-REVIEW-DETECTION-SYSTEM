import { useState, useEffect } from 'react'
import { motion, useScroll, useTransform } from 'framer-motion'
import Header from './components/Header'
import ReviewForm from './components/ReviewForm'
import Toast from './components/Toast'
import { checkApiAvailability } from './utils/api'

export default function App() {
  const { scrollYProgress } = useScroll()
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })
  const [apiStatus, setApiStatus] = useState({ available: false, checked: false })
  const [toast, setToast] = useState(null)

  // Advanced scroll-based transformations
  const backgroundY = useTransform(scrollYProgress, [0, 1], ['0%', '50%'])
  const textY = useTransform(scrollYProgress, [0, 1], ['0%', '150%'])
  const opacity = useTransform(scrollYProgress, [0, 0.3], [1, 0])

  // Mouse tracking for interactive effects
  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePosition({ x: e.clientX, y: e.clientY })
    }
    window.addEventListener('mousemove', handleMouseMove)
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [])

  // Check API availability on mount
  useEffect(() => {
    const checkApi = async () => {
      try {
        const status = await checkApiAvailability()
        setApiStatus({ ...status, checked: true })
        if (!status.available) {
          showToast('API connection failed. Please ensure the backend is running.', 'error')
        }
      } catch (error) {
        setApiStatus({ available: false, checked: true })
        showToast('Unable to connect to API', 'error')
      }
    }
    checkApi()
  }, [])

  const showToast = (message, type = 'info') => {
    setToast({ message, type })
    setTimeout(() => setToast(null), 5000)
  }

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Dynamic Gradient Background */}
      <div className="fixed inset-0 bg-gradient-to-br from-purple-50 via-white to-blue-50 dark:from-gray-950 dark:via-purple-950/30 dark:to-gray-900">
        {/* Animated Mesh Background */}
        <motion.div 
          className="absolute inset-0 opacity-20 dark:opacity-30"
          style={{
            background: `radial-gradient(circle at ${mousePosition.x}px ${mousePosition.y}px, rgba(168, 85, 247, 0.15) 0%, transparent 50%)`
          }}
        />
        
        {/* Premium Grid Pattern */}
        <motion.div
          style={{ y: backgroundY }}
          className="absolute inset-0"
        >
          <div className="absolute inset-0 opacity-[0.03] dark:opacity-[0.05]" 
            style={{
              backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%239333ea' fill-opacity='1'%3E%3Cpath d='M30 0l25.98 15v30L30 60 4.02 45V15L30 0z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`
            }}
          />
          
          {/* Floating Elements */}
          {[...Array(15)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute"
              initial={{
                x: Math.random() * (typeof window !== 'undefined' ? window.innerWidth : 1000),
                y: Math.random() * (typeof window !== 'undefined' ? window.innerHeight : 1000),
              }}
              animate={{
                x: Math.random() * (typeof window !== 'undefined' ? window.innerWidth : 1000),
                y: Math.random() * (typeof window !== 'undefined' ? window.innerHeight : 1000),
              }}
              transition={{
                duration: Math.random() * 30 + 40,
                repeat: Infinity,
                repeatType: "reverse",
                ease: "linear"
              }}
            >
              <div 
                className="w-2 h-2 bg-gradient-to-r from-purple-500 to-pink-500 dark:from-purple-400 dark:to-pink-400 rounded-full opacity-30 blur-sm" 
              />
            </motion.div>
          ))}
        </motion.div>

        {/* Gradient Orbs */}
        <motion.div 
          className="absolute top-1/4 left-1/6 w-[500px] h-[500px] bg-gradient-to-r from-purple-400/20 via-pink-400/20 to-blue-400/20 dark:from-purple-600/30 dark:via-pink-600/30 dark:to-blue-600/30 rounded-full filter blur-3xl"
          animate={{ 
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.15, 0.3],
            rotate: [0, 360]
          }}
          transition={{ 
            duration: 30,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
        <motion.div 
          className="absolute bottom-1/4 right-1/6 w-[400px] h-[400px] bg-gradient-to-r from-blue-400/20 via-cyan-400/20 to-teal-400/20 dark:from-blue-600/30 dark:via-cyan-600/30 dark:to-teal-600/30 rounded-full filter blur-3xl"
          animate={{ 
            scale: [1.2, 1, 1.2],
            opacity: [0.2, 0.35, 0.2],
            rotate: [360, 0]
          }}
          transition={{ 
            duration: 25,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
      </div>

      {/* Scroll Progress Bar */}
      <motion.div
        className="fixed top-0 left-0 right-0 h-1 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 origin-left z-50 shadow-lg shadow-purple-500/50"
        style={{ scaleX: scrollYProgress }}
      />

      {/* Main Content */}
      <div className="relative z-10">
        <Header apiStatus={apiStatus} />
        
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1.2 }}
          className="max-w-7xl mx-auto px-4 py-12 md:py-20"
        >
          {/* Hero Content */}
          <motion.div
            style={{ y: textY, opacity }}
            className="text-center mb-16 md:mb-24"
          >
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="relative"
            >
              {/* Status Badge */}
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.6, delay: 0.3 }}
                className={`inline-flex items-center gap-2 px-4 py-2 mb-6 rounded-full text-sm font-medium backdrop-blur-xl border ${
                  apiStatus.available
                    ? 'bg-green-500/10 dark:bg-green-500/20 border-green-500/30 text-green-700 dark:text-green-400'
                    : 'bg-gray-500/10 dark:bg-gray-500/20 border-gray-500/30 text-gray-700 dark:text-gray-400'
                }`}
              >
                <span className={`w-2 h-2 rounded-full ${apiStatus.available ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></span>
                {apiStatus.checked ? (apiStatus.available ? 'AI Engine Active' : 'Connecting...') : 'Initializing...'}
              </motion.div>

              {/* Main Title */}
              <motion.h1
                className="text-5xl md:text-7xl lg:text-8xl font-black mb-6 leading-none"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.4 }}
              >
                <span className="block gradient-text animate-gradient">
                  Review
                </span>
                <span className="block text-gray-900 dark:text-gray-100">
                  Authenticator
                </span>
              </motion.h1>

              {/* Subtitle */}
              <motion.p
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.6 }}
                className="text-lg md:text-xl text-gray-600 dark:text-gray-400 mb-8 max-w-3xl mx-auto leading-relaxed"
              >
                Advanced AI-powered detection system to identify authentic reviews from fake ones.
                <br className="hidden md:block" />
                Get instant analysis with{' '}
                <motion.span 
                  className="font-bold text-green-600 dark:text-green-400 px-2 py-1 bg-green-500/10 dark:bg-green-500/20 rounded-lg border border-green-500/20"
                  whileHover={{ scale: 1.05 }}
                >
                  confidence scores
                </motion.span>{' '}
                and{' '}
                <motion.span 
                  className="font-bold text-purple-600 dark:text-purple-400 px-2 py-1 bg-purple-500/10 dark:bg-purple-500/20 rounded-lg border border-purple-500/20"
                  whileHover={{ scale: 1.05 }}
                >
                  detailed explanations
                </motion.span>
              </motion.p>
            </motion.div>
          </motion.div>

          {/* Main Form Container */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.8 }}
            className="relative mb-16 md:mb-24"
          >
            {/* Glow Effect */}
            <div className="absolute -inset-1 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 rounded-3xl blur-lg opacity-20 dark:opacity-30 animate-pulse-slow"></div>
            
            <motion.div
              whileHover={{ scale: 1.01 }}
              transition={{ type: "spring", stiffness: 300, damping: 20 }}
              className="relative glass-card p-6 md:p-12 shadow-2xl"
            >
              <ReviewForm showToast={showToast} />
            </motion.div>
          </motion.div>

          {/* Features Grid */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 1.2 }}
            className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16 md:mb-24"
          >
            {[
              {
                icon: "⚡",
                title: "Instant Analysis",
                description: "Get results in milliseconds with our optimized ML pipeline",
                gradient: "from-yellow-500 to-orange-500"
              },
              {
                icon: "🎯",
                title: "High Accuracy",
                description: "Multi-model ensemble for maximum prediction reliability",
                gradient: "from-purple-500 to-pink-500"
              },
              {
                icon: "🔍",
                title: "Deep Insights",
                description: "SHAP explanations show exactly why reviews are classified",
                gradient: "from-blue-500 to-cyan-500"
              },
              {
                icon: "📊",
                title: "Visual Reports",
                description: "Interactive charts and confidence visualizations",
                gradient: "from-green-500 to-teal-500"
              }
            ].map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 1.4 + index * 0.1 }}
                whileHover={{ 
                  y: -8,
                  transition: { type: "spring", stiffness: 400 }
                }}
                className="group relative"
              >
                <div className={`absolute inset-0 bg-gradient-to-r ${feature.gradient} rounded-2xl blur opacity-0 group-hover:opacity-20 dark:group-hover:opacity-30 transition-opacity duration-500`}></div>
                <div className="relative glass-card p-6 text-center group-hover:border-purple-500/30 transition-all duration-300 h-full">
                  <motion.div
                    className="text-4xl mb-4"
                    whileHover={{ 
                      scale: 1.2, 
                      rotate: [0, -5, 5, 0],
                      transition: { duration: 0.4 }
                    }}
                  >
                    {feature.icon}
                  </motion.div>
                  <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-3">
                    {feature.title}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                    {feature.description}
                  </p>
                </div>
              </motion.div>
            ))}
          </motion.div>

          {/* Stats Section */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 1.8 }}
            className="grid md:grid-cols-3 gap-6 mb-16"
          >
            {[
              { number: "5", label: "ML Models", suffix: "+" },
              { number: "95", label: "Accuracy Rate", suffix: "%" },
              { number: "<1", label: "Response Time", suffix: "s" }
            ].map((stat, index) => (
              <motion.div
                key={index}
                whileHover={{ scale: 1.05 }}
                className="glass-card p-8 text-center"
              >
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ duration: 0.8, delay: 2 + index * 0.1, type: "spring", stiffness: 200 }}
                  className="text-4xl md:text-5xl font-black gradient-text mb-2"
                >
                  {stat.number}{stat.suffix}
                </motion.div>
                <p className="text-gray-600 dark:text-gray-400 font-medium">{stat.label}</p>
              </motion.div>
            ))}
          </motion.div>
        </motion.div>

        {/* Footer */}
        <motion.footer
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 2.2 }}
          className="py-12 border-t border-gray-200 dark:border-gray-800"
        >
          <div className="max-w-7xl mx-auto px-4 text-center">
            <motion.div
              whileHover={{ scale: 1.02 }}
              className="inline-block glass-card p-6 rounded-2xl"
            >
              <p className="text-gray-600 dark:text-gray-400 mb-4">Built with</p>
              <div className="flex justify-center items-center flex-wrap gap-4">
                {[
                  { name: "React", icon: "⚛️" },
                  { name: "Tailwind", icon: "🎨" },
                  { name: "Framer", icon: "🎭" },
                  { name: "Flask", icon: "🐍" }
                ].map((tech, index) => (
                  <motion.div
                    key={index}
                    whileHover={{ scale: 1.1, y: -3 }}
                    className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-gray-700 dark:text-gray-300"
                  >
                    <span>{tech.icon}</span>
                    <span className="font-semibold text-sm">{tech.name}</span>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </div>
        </motion.footer>
      </div>

      {/* Mouse Follower */}
      <motion.div
        className="hidden md:block fixed pointer-events-none z-50 w-4 h-4 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full opacity-40 blur-sm"
        animate={{
          x: mousePosition.x - 8,
          y: mousePosition.y - 8,
        }}
        transition={{ type: "spring", stiffness: 500, damping: 28 }}
      />

      {/* Toast Notifications */}
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
    </div>
  )
}