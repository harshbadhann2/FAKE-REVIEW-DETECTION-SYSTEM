// import { motion } from "framer-motion"

// export default function ProbabilityBar({ label, value }) {
//   const pct = Math.round(value * 100)
//   const isReal = label.toLowerCase().includes("real") || label.toLowerCase().includes("authentic")

//   return (
//     <motion.div 
//       initial={{ opacity: 0, y: 20 }}
//       animate={{ opacity: 1, y: 0 }}
//       transition={{ duration: 0.6 }}
//       className="w-full group"
//     >
//       {/* Premium Label + Percentage Container */}
//       <div className="flex justify-between items-center mb-4">
//         {/* Premium Label with Status Indicator */}
//         <div className="flex items-center gap-3">
//           <motion.div
//             initial={{ scale: 0, rotate: -180 }}
//             animate={{ scale: 1, rotate: 0 }}
//             transition={{ delay: 0.2, type: "spring", stiffness: 500 }}
//             className={`relative w-4 h-4 rounded-full shadow-lg ${
//               isReal
//                 ? "bg-gradient-to-r from-emerald-400 to-green-500 shadow-emerald-500/50"
//                 : "bg-gradient-to-r from-red-400 to-pink-500 shadow-red-500/50"
//             }`}
//           >
//             <motion.div
//               className="absolute inset-0 rounded-full bg-current opacity-60 blur-sm"
//               animate={{ scale: [1, 1.4, 1] }}
//               transition={{ duration: 2, repeat: Infinity }}
//             />
//           </motion.div>
          
//           <span className="text-gray-200 font-bold tracking-wide text-base uppercase">
//             {label}
//           </span>
//         </div>

//         {/* Premium Percentage Display */}
//         <motion.div
//           initial={{ opacity: 0, scale: 0.5 }}
//           animate={{ opacity: 1, scale: 1 }}
//           transition={{ delay: 0.4, type: "spring", stiffness: 300 }}
//           className="relative"
//         >
//           <motion.div
//             className={`px-4 py-2 rounded-xl backdrop-blur-xl border font-black text-xl shadow-2xl ${
//               isReal
//                 ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/30 shadow-emerald-500/20"
//                 : "text-red-400 bg-red-500/10 border-red-500/30 shadow-red-500/20"
//             }`}
//             whileHover={{ 
//               scale: 1.1,
//               boxShadow: isReal 
//                 ? "0 0 25px rgba(16, 185, 129, 0.5)"
//                 : "0 0 25px rgba(239, 68, 68, 0.5)"
//             }}
//           >
//             {pct}%
//           </motion.div>
          
//           {/* Premium Glow Effect */}
//           <motion.div
//             className={`absolute inset-0 rounded-xl blur-lg opacity-0 group-hover:opacity-40 ${
//               isReal ? "bg-emerald-400" : "bg-red-400"
//             }`}
//             animate={{ opacity: [0, 0.3, 0] }}
//             transition={{ duration: 3, repeat: Infinity }}
//           />
//         </motion.div>
//       </div>

//       {/* Premium Track Container */}
//       <motion.div
//         initial={{ opacity: 0, scale: 0.9 }}
//         animate={{ opacity: 1, scale: 1 }}
//         transition={{ delay: 0.3, duration: 0.8 }}
//         className="relative group"
//       >
//         {/* Premium Outer Glow Border */}
//         <div className={`absolute -inset-1 rounded-2xl blur-sm opacity-30 group-hover:opacity-50 transition-opacity duration-500 ${
//           isReal
//             ? "bg-gradient-to-r from-emerald-400 via-green-500 to-emerald-600"
//             : "bg-gradient-to-r from-red-400 via-pink-500 to-red-600"
//         }`}></div>
        
//         {/* Main Track */}
//         <div className="relative h-8 backdrop-blur-2xl bg-gray-800/70 border border-gray-600/50 rounded-2xl overflow-hidden shadow-2xl">
//           {/* Premium Progress Markers */}
//           <div className="absolute inset-0 flex justify-between items-center px-3 pointer-events-none z-10">
//             {[25, 50, 75].map((marker) => (
//               <motion.div
//                 key={marker}
//                 initial={{ opacity: 0 }}
//                 animate={{ opacity: pct >= marker ? 0.8 : 0.3 }}
//                 transition={{ delay: 0.8, duration: 0.5 }}
//                 className="w-px h-4 bg-gray-400/60 shadow-sm"
//               />
//             ))}
//           </div>

//           {/* Animated Fill Bar */}
//           <motion.div
//             initial={{ width: 0, opacity: 0 }}
//             animate={{ width: `${pct}%`, opacity: 1 }}
//             transition={{ 
//               duration: 1.5, 
//               ease: "easeOut",
//               opacity: { delay: 0.5, duration: 0.4 }
//             }}
//             className={`relative h-full rounded-2xl overflow-hidden shadow-inner ${
//               isReal
//                 ? "bg-gradient-to-r from-emerald-400 via-green-500 to-emerald-600"
//                 : "bg-gradient-to-r from-red-400 via-pink-500 to-red-600"
//             }`}
//           >
//             {/* Premium Shine Effect */}
//             <motion.div
//               className="absolute inset-0 bg-gradient-to-r from-transparent via-white/40 to-transparent -skew-x-12"
//               animate={{ x: ["-150%", "150%"] }}
//               transition={{ 
//                 duration: 2.5,
//                 delay: 1.2,
//                 repeat: Infinity,
//                 repeatDelay: 4
//               }}
//             />
            
//             {/* Premium Inner Depth Effect */}
//             <div className="absolute inset-0 bg-gradient-to-b from-white/20 via-transparent to-black/20 rounded-2xl"></div>
            
//             {/* Premium End Glow */}
//             <motion.div
//               className={`absolute right-0 top-0 bottom-0 w-2 ${
//                 isReal ? "bg-emerald-300" : "bg-red-300"
//               } blur-sm opacity-80`}
//               animate={{ opacity: [0.6, 1, 0.6] }}
//               transition={{ duration: 1.5, repeat: Infinity }}
//             />
//           </motion.div>

//           {/* Premium Completion Badge */}
//           {pct >= 95 && (
//             <motion.div
//               initial={{ scale: 0, rotate: -180 }}
//               animate={{ scale: 1, rotate: 0 }}
//               transition={{ delay: 1.5, type: "spring", stiffness: 600 }}
//               className={`absolute right-2 top-1/2 -translate-y-1/2 w-6 h-6 rounded-full border-2 border-gray-900 shadow-xl ${
//                 isReal
//                   ? "bg-gradient-to-r from-emerald-300 to-green-400"
//                   : "bg-gradient-to-r from-red-300 to-pink-400"
//               }`}
//             >
//               <motion.div
//                 className="absolute inset-0 rounded-full bg-current opacity-50 blur-sm"
//                 animate={{ scale: [1, 1.6, 1] }}
//                 transition={{ duration: 2, repeat: Infinity }}
//               />
//               <div className="absolute inset-2 rounded-full bg-white/90"></div>
//             </motion.div>
//           )}
//         </div>
//       </motion.div>

//       {/* Premium Confidence Status */}
//       <motion.div
//         initial={{ opacity: 0 }}
//         animate={{ opacity: 1 }}
//         transition={{ delay: 1.3 }}
//         className="mt-3 text-center"
//       >
//         <span className={`text-xs font-black tracking-widest px-3 py-1 rounded-lg backdrop-blur-lg border ${
//           pct >= 90 
//             ? isReal 
//               ? "text-emerald-400 bg-emerald-500/10 border-emerald-500/30" 
//               : "text-red-400 bg-red-500/10 border-red-500/30"
//             : pct >= 70 
//               ? "text-amber-400 bg-amber-500/10 border-amber-500/30"
//               : "text-gray-500 bg-gray-500/10 border-gray-500/30"
//         }`}>
//           {pct >= 90 ? "MAXIMUM CONFIDENCE" :
//            pct >= 70 ? "HIGH CONFIDENCE" :
//            pct >= 50 ? "MODERATE CONFIDENCE" :
//            "LOW CONFIDENCE"}
//         </span>
//       </motion.div>
//     </motion.div>
//   )
// }









import { motion } from "framer-motion"

export default function ProbabilityBar({ label, value, color = "purple" }) {
  const pct = Math.round(value * 100)
  
  // Color mappings
  const colors = {
    green: {
      gradient: "from-green-500 to-emerald-500",
      bg: "bg-green-500/10 dark:bg-green-500/20",
      border: "border-green-500/30",
      text: "text-green-700 dark:text-green-400",
      glow: "shadow-green-500/50"
    },
    red: {
      gradient: "from-red-500 to-pink-500",
      bg: "bg-red-500/10 dark:bg-red-500/20",
      border: "border-red-500/30",
      text: "text-red-700 dark:text-red-400",
      glow: "shadow-red-500/50"
    },
    purple: {
      gradient: "from-purple-500 to-blue-500",
      bg: "bg-purple-500/10 dark:bg-purple-500/20",
      border: "border-purple-500/30",
      text: "text-purple-700 dark:text-purple-400",
      glow: "shadow-purple-500/50"
    }
  }

  const colorScheme = colors[color] || colors.purple

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="w-full"
    >
      {/* Label and Percentage */}
      <div className="flex justify-between items-center mb-2">
        <div className="flex items-center gap-2">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.1, type: "spring", stiffness: 300 }}
            className={`w-3 h-3 rounded-full bg-gradient-to-r ${colorScheme.gradient}`}
          />
          <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">
            {label}
          </span>
        </div>

        <motion.span
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
          className={`text-sm font-bold ${colorScheme.text}`}
        >
          {pct}%
        </motion.span>
      </div>

      {/* Progress Bar Container */}
      <div className="relative h-3 bg-gray-200 dark:bg-gray-800 rounded-full overflow-hidden">
        {/* Progress markers (25%, 50%, 75%) */}
        <div className="absolute inset-0 flex justify-between items-center px-[25%]">
          {[1, 2].map((i) => (
            <div
              key={i}
              className="w-px h-2 bg-gray-300 dark:bg-gray-700 opacity-50"
            />
          ))}
        </div>

        {/* Animated fill */}
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ 
            duration: 1,
            ease: "easeOut",
            delay: 0.3
          }}
          className={`relative h-full bg-gradient-to-r ${colorScheme.gradient} rounded-full`}
        >
          {/* Shine effect */}
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
            animate={{ x: ["-100%", "200%"] }}
            transition={{ 
              duration: 2,
              delay: 1,
              repeat: Infinity,
              repeatDelay: 3
            }}
          />

          {/* Inner highlight */}
          <div className="absolute inset-0 bg-gradient-to-b from-white/20 to-transparent rounded-full" />
        </motion.div>

        {/* End glow for high values */}
        {pct > 80 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.6 }}
            transition={{ delay: 1, duration: 0.5 }}
            className={`absolute right-0 top-0 bottom-0 w-8 bg-gradient-to-l ${colorScheme.gradient} blur-sm`}
          />
        )}
      </div>

      {/* Confidence level indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
        className="mt-1.5 text-right"
      >
        <span className={`text-xs font-medium ${
          pct >= 85 
            ? colorScheme.text
            : pct >= 70 
            ? "text-yellow-600 dark:text-yellow-400"
            : "text-gray-500 dark:text-gray-400"
        }`}>
          {pct >= 85 ? "High" : pct >= 70 ? "Medium" : "Low"} certainty
        </span>
      </motion.div>
    </motion.div>
  )
}