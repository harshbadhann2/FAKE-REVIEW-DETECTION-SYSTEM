// import { useEffect, useState } from 'react'
// import { motion, AnimatePresence } from 'framer-motion'
// import { CheckCircle, XCircle, X, Shield, AlertTriangle } from 'lucide-react'

// export default function Toast({ show, type = 'error', message, onClose }) {
//   const [progress, setProgress] = useState(100)

//   useEffect(() => {
//     if (!show) return
//     setProgress(100)

//     const duration = 4000 // Extended duration for premium feel
//     const interval = 20
//     let elapsed = 0

//     const timer = setInterval(() => {
//       elapsed += interval
//       setProgress(100 - (elapsed / duration) * 100)
//       if (elapsed >= duration) {
//         clearInterval(timer)
//         onClose()
//       }
//     }, interval)

//     return () => clearInterval(timer)
//   }, [show, onClose])

//   const getToastConfig = () => {
//     if (type === 'error') {
//       return {
//         icon: XCircle,
//         bgGradient: 'from-red-500/20 via-red-600/20 to-red-700/20',
//         borderGradient: 'from-red-500/60 via-red-600/60 to-red-700/60',
//         glowColor: 'red-500',
//         progressGradient: 'from-red-400 to-red-600',
//         statusIcon: '❌',
//         title: 'SYSTEM ERROR',
//         accentColor: 'text-red-300'
//       }
//     } else {
//       return {
//         icon: CheckCircle,
//         bgGradient: 'from-emerald-500/20 via-green-600/20 to-emerald-700/20',
//         borderGradient: 'from-emerald-500/60 via-green-600/60 to-emerald-700/60',
//         glowColor: 'emerald-500',
//         progressGradient: 'from-emerald-400 to-green-600',
//         statusIcon: '✅',
//         title: 'MISSION SUCCESS',
//         accentColor: 'text-emerald-300'
//       }
//     }
//   }

//   const config = getToastConfig()
//   const IconComponent = config.icon

//   return (
//     <AnimatePresence mode="wait">
//       {show && (
//         <motion.div
//           initial={{ x: 400, opacity: 0, scale: 0.8, rotateY: -90 }}
//           animate={{ x: 0, opacity: 1, scale: 1, rotateY: 0 }}
//           exit={{ 
//             x: 400, 
//             opacity: 0, 
//             scale: 0.8, 
//             rotateY: 90,
//             transition: { duration: 0.4, ease: "easeIn" }
//           }}
//           transition={{ 
//             type: "spring",
//             stiffness: 300,
//             damping: 25,
//             duration: 0.6
//           }}
//           className="fixed bottom-8 right-8 z-[100]"
//         >
//           {/* Premium Container with Advanced Effects */}
//           <motion.div
//             whileHover={{ 
//               scale: 1.05,
//               rotateY: 5,
//               transition: { type: "spring", stiffness: 400 }
//             }}
//             className="relative group"
//           >
//             {/* Dynamic Glow Effect */}
//             <motion.div
//               className={`absolute -inset-1 bg-gradient-to-r ${config.borderGradient} rounded-2xl blur-md opacity-60 group-hover:opacity-80`}
//               animate={{
//                 scale: [1, 1.1, 1],
//                 opacity: [0.6, 0.8, 0.6]
//               }}
//               transition={{
//                 duration: 2,
//                 repeat: Infinity,
//                 ease: "easeInOut"
//               }}
//             />

//             {/* Main Toast Container */}
//             <div className={`relative backdrop-blur-2xl bg-gradient-to-r ${config.bgGradient} 
//                            border border-gray-700/30 rounded-2xl shadow-2xl overflow-hidden w-96`}>
              
//               {/* Premium Header */}
//               <div className="relative px-6 pt-5 pb-3">
//                 {/* Corner Accents */}
//                 <div className="absolute top-3 left-3 w-4 h-4 border-l-2 border-t-2 border-amber-500/60 rounded-tl-lg" />
//                 <div className="absolute top-3 right-3 w-4 h-4 border-r-2 border-t-2 border-amber-500/60 rounded-tr-lg" />

//                 {/* Header Content */}
//                 <div className="flex items-center justify-between">
//                   <div className="flex items-center gap-3">
//                     {/* Animated Status Icon */}
//                     <motion.div
//                       initial={{ scale: 0, rotate: -180 }}
//                       animate={{ scale: 1, rotate: 0 }}
//                       transition={{ 
//                         type: "spring", 
//                         stiffness: 500, 
//                         delay: 0.2 
//                       }}
//                       className="relative"
//                     >
//                       <motion.div
//                         animate={{ 
//                           boxShadow: [
//                             `0 0 0px rgba(${type === 'error' ? '239, 68, 68' : '16, 185, 129'}, 0)`,
//                             `0 0 20px rgba(${type === 'error' ? '239, 68, 68' : '16, 185, 129'}, 0.6)`,
//                             `0 0 0px rgba(${type === 'error' ? '239, 68, 68' : '16, 185, 129'}, 0)`
//                           ]
//                         }}
//                         transition={{ duration: 2, repeat: Infinity }}
//                         className={`w-10 h-10 rounded-full bg-gradient-to-r ${config.progressGradient} 
//                                    flex items-center justify-center shadow-lg`}
//                       >
//                         <IconComponent className="w-6 h-6 text-white" />
//                       </motion.div>
//                     </motion.div>

//                     {/* Title and Status */}
//                     <div>
//                       <motion.div
//                         initial={{ opacity: 0, x: -20 }}
//                         animate={{ opacity: 1, x: 0 }}
//                         transition={{ duration: 0.5, delay: 0.3 }}
//                         className="flex items-center gap-2"
//                       >
//                         <h4 className={`font-black text-sm tracking-wider uppercase ${config.accentColor}`}>
//                           {config.title}
//                         </h4>
//                         <motion.span
//                           animate={{ rotate: [0, 10, -10, 0] }}
//                           transition={{ duration: 2, repeat: Infinity, delay: 1 }}
//                           className="text-lg"
//                         >
//                           {config.statusIcon}
//                         </motion.span>
//                       </motion.div>
//                       <motion.div
//                         initial={{ opacity: 0, x: -20 }}
//                         animate={{ opacity: 1, x: 0 }}
//                         transition={{ duration: 0.5, delay: 0.4 }}
//                         className="flex items-center gap-2 mt-1"
//                       >
//                         <div className={`w-2 h-0.5 bg-gradient-to-r ${config.progressGradient} rounded-full`} />
//                         <span className="text-xs text-gray-400 font-medium tracking-wide uppercase">
//                           Neural Response
//                         </span>
//                       </motion.div>
//                     </div>
//                   </div>

//                   {/* Premium Close Button */}
//                   <motion.button
//                     whileHover={{ 
//                       scale: 1.1, 
//                       rotate: 90,
//                       backgroundColor: "rgba(255, 255, 255, 0.1)"
//                     }}
//                     whileTap={{ scale: 0.9 }}
//                     onClick={onClose}
//                     className="relative p-2 rounded-xl bg-gray-800/30 border border-gray-600/30 
//                                text-gray-400 hover:text-white transition-all duration-300 backdrop-blur-sm"
//                   >
//                     <X className="w-4 h-4" />
//                     {/* Button Glow */}
//                     <div className="absolute inset-0 rounded-xl bg-white/10 opacity-0 hover:opacity-100 transition-opacity blur-sm" />
//                   </motion.button>
//                 </div>
//               </div>

//               {/* Message Content */}
//               <motion.div
//                 initial={{ opacity: 0, y: 20 }}
//                 animate={{ opacity: 1, y: 0 }}
//                 transition={{ duration: 0.6, delay: 0.5 }}
//                 className="px-6 pb-5"
//               >
//                 <div className="bg-gray-950/40 border border-gray-700/30 rounded-xl p-4 backdrop-blur-sm">
//                   <p className="text-gray-200 text-sm leading-relaxed font-medium">
//                     {message}
//                   </p>
                  
//                   {/* Status Indicator Bar */}
//                   <div className="flex items-center gap-2 mt-3">
//                     <Shield className="w-3 h-3 text-gray-500" />
//                     <div className="flex-1 h-1 bg-gray-800 rounded-full overflow-hidden">
//                       <motion.div
//                         initial={{ width: "0%" }}
//                         animate={{ width: "100%" }}
//                         transition={{ duration: 1, delay: 0.6 }}
//                         className={`h-full bg-gradient-to-r ${config.progressGradient} rounded-full`}
//                       />
//                     </div>
//                     <span className="text-xs text-gray-500 font-mono">100%</span>
//                   </div>
//                 </div>
//               </motion.div>

//               {/* Enhanced Progress Bar */}
//               <div className="relative h-2 bg-gray-800/50 overflow-hidden">
//                 <motion.div
//                   initial={{ width: '100%' }}
//                   animate={{ width: `${progress}%` }}
//                   transition={{ duration: 0.05, ease: "linear" }}
//                   className={`h-full bg-gradient-to-r ${config.progressGradient} relative overflow-hidden`}
//                 >
//                   {/* Animated Progress Shine */}
//                   <motion.div
//                     animate={{
//                       x: ['-100%', '200%']
//                     }}
//                     transition={{
//                       duration: 2,
//                       repeat: Infinity,
//                       ease: "linear"
//                     }}
//                     className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent skew-x-12"
//                   />
//                 </motion.div>
//               </div>

//               {/* Premium Bottom Accent */}
//               <div className={`h-1 bg-gradient-to-r ${config.borderGradient} opacity-60`} />
//             </div>

//             {/* Floating Particles Effect */}
//             {[...Array(6)].map((_, i) => (
//               <motion.div
//                 key={i}
//                 className={`absolute w-1 h-1 bg-${config.glowColor} rounded-full opacity-40`}
//                 initial={{
//                   x: Math.random() * 100 - 50,
//                   y: Math.random() * 100 - 50,
//                   scale: 0
//                 }}
//                 animate={{
//                   x: Math.random() * 200 - 100,
//                   y: Math.random() * 200 - 100,
//                   scale: [0, 1, 0],
//                   opacity: [0, 0.6, 0]
//                 }}
//                 transition={{
//                   duration: Math.random() * 3 + 2,
//                   repeat: Infinity,
//                   delay: Math.random() * 2
//                 }}
//               />
//             ))}
//           </motion.div>
//         </motion.div>
//       )}
//     </AnimatePresence>
//   )
// }























import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle, XCircle, X, AlertCircle, Info } from 'lucide-react'

export default function Toast({ message, type = 'info', onClose }) {
  const [progress, setProgress] = useState(100)
  const show = !!message

  useEffect(() => {
    if (!show) return
    setProgress(100)

    const duration = 5000
    const interval = 20
    let elapsed = 0

    const timer = setInterval(() => {
      elapsed += interval
      setProgress(100 - (elapsed / duration) * 100)
      if (elapsed >= duration) {
        clearInterval(timer)
        onClose?.()
      }
    }, interval)

    return () => clearInterval(timer)
  }, [show, message, onClose])

  const getToastConfig = () => {
    switch (type) {
      case 'success':
        return {
          icon: CheckCircle,
          bgClass: 'bg-green-500/10 dark:bg-green-500/20',
          borderClass: 'border-green-500/30',
          textClass: 'text-green-700 dark:text-green-400',
          iconClass: 'text-green-600 dark:text-green-400',
          progressClass: 'bg-green-500',
          glowClass: 'shadow-green-500/50'
        }
      case 'error':
        return {
          icon: XCircle,
          bgClass: 'bg-red-500/10 dark:bg-red-500/20',
          borderClass: 'border-red-500/30',
          textClass: 'text-red-700 dark:text-red-400',
          iconClass: 'text-red-600 dark:text-red-400',
          progressClass: 'bg-red-500',
          glowClass: 'shadow-red-500/50'
        }
      case 'warning':
        return {
          icon: AlertCircle,
          bgClass: 'bg-yellow-500/10 dark:bg-yellow-500/20',
          borderClass: 'border-yellow-500/30',
          textClass: 'text-yellow-700 dark:text-yellow-400',
          iconClass: 'text-yellow-600 dark:text-yellow-400',
          progressClass: 'bg-yellow-500',
          glowClass: 'shadow-yellow-500/50'
        }
      default: // info
        return {
          icon: Info,
          bgClass: 'bg-blue-500/10 dark:bg-blue-500/20',
          borderClass: 'border-blue-500/30',
          textClass: 'text-blue-700 dark:text-blue-400',
          iconClass: 'text-blue-600 dark:text-blue-400',
          progressClass: 'bg-blue-500',
          glowClass: 'shadow-blue-500/50'
        }
    }
  }

  const config = getToastConfig()
  const IconComponent = config.icon

  return (
    <AnimatePresence mode="wait">
      {show && (
        <motion.div
          initial={{ x: 400, opacity: 0, scale: 0.9 }}
          animate={{ x: 0, opacity: 1, scale: 1 }}
          exit={{ 
            x: 400, 
            opacity: 0, 
            scale: 0.9,
            transition: { duration: 0.3 }
          }}
          transition={{ 
            type: "spring",
            stiffness: 300,
            damping: 25
          }}
          className="fixed top-4 right-4 z-[100] max-w-md"
        >
          <motion.div
            whileHover={{ scale: 1.02 }}
            className="relative"
          >
            {/* Glow effect */}
            <div className={`absolute -inset-1 ${config.bgClass} blur-lg opacity-50 rounded-2xl`} />

            {/* Main content */}
            <div className={`relative glass-card ${config.borderClass} border shadow-xl overflow-hidden`}>
              <div className="p-4">
                <div className="flex items-start gap-3">
                  {/* Icon */}
                  <motion.div
                    initial={{ scale: 0, rotate: -180 }}
                    animate={{ scale: 1, rotate: 0 }}
                    transition={{ 
                      type: "spring", 
                      stiffness: 400, 
                      delay: 0.1 
                    }}
                    className={`flex-shrink-0 ${config.iconClass}`}
                  >
                    <IconComponent size={24} strokeWidth={2.5} />
                  </motion.div>

                  {/* Message */}
                  <motion.div
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                    className="flex-1 min-w-0"
                  >
                    <p className={`text-sm font-medium leading-relaxed ${config.textClass}`}>
                      {message}
                    </p>
                  </motion.div>

                  {/* Close button */}
                  <motion.button
                    whileHover={{ scale: 1.1, rotate: 90 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={onClose}
                    className="flex-shrink-0 p-1 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-800 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition-colors"
                  >
                    <X size={16} />
                  </motion.button>
                </div>
              </div>

              {/* Progress bar */}
              <div className="h-1 bg-gray-200 dark:bg-gray-800 overflow-hidden">
                <motion.div
                  initial={{ width: '100%' }}
                  animate={{ width: `${progress}%` }}
                  className={`h-full ${config.progressClass}`}
                  style={{ transition: 'width 0.02s linear' }}
                >
                  {/* Shine effect */}
                  <motion.div
                    animate={{ x: ['-100%', '200%'] }}
                    transition={{
                      duration: 1.5,
                      repeat: Infinity,
                      ease: "linear"
                    }}
                    className="h-full w-full bg-gradient-to-r from-transparent via-white/30 to-transparent"
                  />
                </motion.div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}