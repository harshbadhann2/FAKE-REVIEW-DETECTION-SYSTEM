// import DarkModeToggle from "./DarkModeToggle";
// import { ShoppingCart, User, Search, Shield } from "lucide-react";
// import { motion } from "framer-motion";

// export default function Header() {
//   return (
//     <motion.header 
//       initial={{ y: -100, opacity: 0 }}
//       animate={{ y: 0, opacity: 1 }}
//       transition={{ duration: 0.8, type: "spring", stiffness: 100 }}
//       className="sticky top-0 z-50 backdrop-blur-2xl bg-gray-900/80 border-b border-gray-700/30 shadow-2xl"
//     >
//       {/* Premium Glow Effect */}
//       <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-amber-500/50 to-transparent"></div>
      
//       <div className="max-w-7xl mx-auto px-4 py-4 flex items-center gap-8">
//         {/* Premium Logo / Brand */}
//         <motion.div 
//           whileHover={{ scale: 1.05 }}
//           whileTap={{ scale: 0.95 }}
//           className="flex items-center gap-3 cursor-pointer group"
//         >
//           <motion.div 
//             className="relative w-12 h-12 rounded-2xl bg-gradient-to-tr from-amber-500 via-orange-500 to-red-500 flex items-center justify-center text-white font-black text-2xl shadow-2xl"
//             whileHover={{ 
//               rotate: [0, -5, 5, 0],
//               boxShadow: "0 0 30px rgba(245, 158, 11, 0.5)"
//             }}
//             transition={{ duration: 0.6 }}
//           >
//             {/* Premium Corner Accents */}
//             <div className="absolute top-1 left-1 w-2 h-2 border-l border-t border-white/60 rounded-tl-sm"></div>
//             <div className="absolute top-1 right-1 w-2 h-2 border-r border-t border-white/60 rounded-tr-sm"></div>
//             <div className="absolute bottom-1 left-1 w-2 h-2 border-l border-b border-white/60 rounded-bl-sm"></div>
//             <div className="absolute bottom-1 right-1 w-2 h-2 border-r border-b border-white/60 rounded-br-sm"></div>
            
//             <Shield size={24} strokeWidth={3} />
//           </motion.div>
          
//           <div className="flex flex-col">
//             <motion.span 
//               className="font-black text-2xl text-transparent bg-gradient-to-r from-amber-400 via-orange-400 to-red-400 bg-clip-text tracking-tight"
//               whileHover={{ scale: 1.05 }}
//             >
//               REVIEW
//             </motion.span>
//             <span className="text-xs text-gray-400 font-bold tracking-widest -mt-1">
//               AUTHENTICATOR
//             </span>
//           </div>
          
//           {/* Premium Status Badge */}
//           <motion.div
//             initial={{ scale: 0 }}
//             animate={{ scale: 1 }}
//             transition={{ delay: 0.5, type: "spring", stiffness: 500 }}
//             className="ml-2 px-2 py-1 bg-gradient-to-r from-amber-500/20 to-orange-500/20 border border-amber-500/30 rounded-md text-xs font-bold text-amber-300"
//           >
//             PRO
//           </motion.div>
//         </motion.div>

//         {/* Premium Search Bar */}
//         <div className="mx-6 flex-1 hidden md:flex">
//           <motion.div 
//             initial={{ opacity: 0, width: 0 }}
//             animate={{ opacity: 1, width: "100%" }}
//             transition={{ delay: 0.3, duration: 0.8 }}
//             className="relative group w-full"
//           >
//             {/* Glowing Border Effect */}
//             <div className="absolute -inset-0.5 bg-gradient-to-r from-amber-500 via-orange-500 to-red-500 rounded-2xl blur opacity-20 group-hover:opacity-40 group-focus-within:opacity-60 transition-opacity duration-500"></div>
            
//             <div className="relative flex w-full rounded-2xl overflow-hidden backdrop-blur-xl bg-gray-800/60 border border-gray-600/30 shadow-2xl">
//               <motion.input
//                 disabled
//                 placeholder="Search Amazon Reviews (PRO Feature)"
//                 className="flex-1 px-6 py-4 bg-transparent text-gray-200 placeholder-gray-400 outline-none font-medium"
//                 whileFocus={{ scale: 1.02 }}
//               />
//               <motion.button 
//                 whileHover={{ scale: 1.1 }}
//                 whileTap={{ scale: 0.95 }}
//                 className="px-6 bg-gradient-to-r from-amber-500 to-orange-500 text-black font-black hover:from-amber-400 hover:to-orange-400 transition-all duration-300 shadow-lg"
//               >
//                 <Search size={20} strokeWidth={2.5} />
//               </motion.button>
//             </div>
//           </motion.div>
//         </div>

//         {/* Premium Right Section */}
//         <div className="flex items-center gap-6">
//           {/* Account Button */}
//           <motion.button
//             whileHover={{ scale: 1.1, y: -2 }}
//             whileTap={{ scale: 0.95 }}
//             className="group relative flex flex-col items-center text-sm text-gray-300 hover:text-amber-400 transition-all duration-300"
//             title="Premium Account"
//           >
//             <motion.div
//               className="relative p-3 rounded-xl backdrop-blur-lg bg-gray-800/40 border border-gray-600/30 group-hover:border-amber-500/50 group-hover:bg-gray-700/50 transition-all duration-300"
//               whileHover={{ boxShadow: "0 0 20px rgba(245, 158, 11, 0.3)" }}
//             >
//               <User size={20} strokeWidth={2.5} />
//               {/* Premium Indicator */}
//               <div className="absolute -top-1 -right-1 w-3 h-3 bg-gradient-to-r from-amber-400 to-orange-500 rounded-full border border-gray-900"></div>
//             </motion.div>
//             <span className="text-xs font-medium hidden sm:block mt-1 group-hover:text-amber-400 transition-colors">
//               ACCOUNT
//             </span>
//           </motion.button>

//           {/* Premium Cart Button */}
//           <motion.button
//             whileHover={{ scale: 1.1, y: -2 }}
//             whileTap={{ scale: 0.95 }}
//             className="group relative flex flex-col items-center text-sm text-gray-300 hover:text-amber-400 transition-all duration-300"
//             title="Premium Cart"
//           >
//             <motion.div
//               className="relative p-3 rounded-xl backdrop-blur-lg bg-gray-800/40 border border-gray-600/30 group-hover:border-amber-500/50 group-hover:bg-gray-700/50 transition-all duration-300"
//               whileHover={{ boxShadow: "0 0 20px rgba(245, 158, 11, 0.3)" }}
//             >
//               <ShoppingCart size={20} strokeWidth={2.5} />
//               {/* Premium Cart Badge */}
//               <motion.span 
//                 initial={{ scale: 0 }}
//                 animate={{ scale: 1 }}
//                 whileHover={{ 
//                   scale: 1.2,
//                   boxShadow: "0 0 15px rgba(245, 158, 11, 0.8)"
//                 }}
//                 className="absolute -top-2 -right-2 bg-gradient-to-r from-red-500 to-pink-500 text-white text-xs font-black px-2 py-1 rounded-full shadow-lg border border-red-400"
//               >
//                 2
//               </motion.span>
//             </motion.div>
//             <span className="text-xs font-medium hidden sm:block mt-1 group-hover:text-amber-400 transition-colors">
//               CART
//             </span>
//           </motion.button>

//           {/* Divider */}
//           <div className="h-12 w-px bg-gradient-to-b from-transparent via-gray-600 to-transparent"></div>

//           {/* Dark Mode Toggle */}
//           <DarkModeToggle />
//         </div>
//       </div>

//       {/* Premium Bottom Accent Line */}
//       <motion.div
//         initial={{ scaleX: 0 }}
//         animate={{ scaleX: 1 }}
//         transition={{ delay: 0.8, duration: 1 }}
//         className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-amber-500/30 to-transparent"
//       />
//     </motion.header>
//   );
// }


















import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Shield, Activity, Menu, X, Github, Info } from "lucide-react"
import DarkModeToggle from "./DarkModeToggle"

export default function Header({ apiStatus }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [showInfo, setShowInfo] = useState(false)

  return (
    <>
      <motion.header 
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, type: "spring", stiffness: 100 }}
        className="sticky top-0 z-50 backdrop-blur-xl bg-white/80 dark:bg-gray-900/80 border-b border-gray-200 dark:border-gray-800 shadow-lg"
      >
        {/* Gradient accent line */}
        <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-purple-500/50 to-transparent"></div>
        
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between gap-4">
            {/* Logo & Brand */}
            <motion.div 
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="flex items-center gap-3 cursor-pointer group"
            >
              {/* Logo Icon */}
              <motion.div 
                className="relative w-12 h-12 rounded-xl bg-gradient-to-br from-purple-600 via-pink-600 to-blue-600 flex items-center justify-center text-white shadow-lg"
                whileHover={{ 
                  rotate: [0, -8, 8, 0],
                  boxShadow: "0 10px 30px rgba(168, 85, 247, 0.4)"
                }}
                transition={{ duration: 0.5 }}
              >
                <Shield size={24} strokeWidth={2.5} />
                
                {/* Corner accents */}
                <div className="absolute top-1 left-1 w-2 h-2 border-l-2 border-t-2 border-white/40 rounded-tl"></div>
                <div className="absolute bottom-1 right-1 w-2 h-2 border-r-2 border-b-2 border-white/40 rounded-br"></div>
              </motion.div>
              
              {/* Brand text */}
              <div className="hidden sm:flex flex-col">
                <motion.span 
                  className="font-black text-xl gradient-text"
                  whileHover={{ scale: 1.05 }}
                >
                  ReviewAuth
                </motion.span>
                <span className="text-xs text-gray-600 dark:text-gray-400 font-semibold tracking-wider -mt-0.5">
                  AI DETECTOR
                </span>
              </div>
              
              {/* Version badge */}
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.4, type: "spring", stiffness: 300 }}
                className="hidden md:block ml-2 px-2 py-1 bg-purple-500/10 dark:bg-purple-500/20 border border-purple-500/30 rounded-md text-xs font-bold text-purple-700 dark:text-purple-400"
              >
                v2.0
              </motion.div>
            </motion.div>

            {/* Center - API Status (desktop) */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 }}
              className="hidden md:flex items-center gap-2 px-4 py-2 rounded-xl glass-card"
            >
              <Activity 
                size={16} 
                className={`${apiStatus?.available ? 'text-green-500' : 'text-gray-400'} ${apiStatus?.available ? 'animate-pulse' : ''}`}
              />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                {apiStatus?.available ? 'API Connected' : 'API Offline'}
              </span>
              {apiStatus?.available && (
                <motion.div
                  className="w-2 h-2 bg-green-500 rounded-full"
                  animate={{ scale: [1, 1.3, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              )}
            </motion.div>

            {/* Right section */}
            <div className="flex items-center gap-3">
              {/* Info button */}
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setShowInfo(true)}
                className="hidden sm:flex items-center gap-2 px-4 py-2 rounded-xl bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 font-medium transition-all"
                title="About"
              >
                <Info size={18} />
                <span className="hidden lg:inline text-sm">About</span>
              </motion.button>

              {/* GitHub link */}
              <motion.a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                className="hidden sm:flex items-center gap-2 px-4 py-2 rounded-xl bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 font-medium transition-all"
                title="View on GitHub"
              >
                <Github size={18} />
                <span className="hidden lg:inline text-sm">GitHub</span>
              </motion.a>

              {/* Dark mode toggle */}
              <DarkModeToggle />

              {/* Mobile menu button */}
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="md:hidden p-2 rounded-xl bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300"
              >
                {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
              </motion.button>
            </div>
          </div>

          {/* Mobile menu */}
          <AnimatePresence>
            {mobileMenuOpen && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.3 }}
                className="md:hidden mt-4 pt-4 border-t border-gray-200 dark:border-gray-800 space-y-3"
              >
                {/* API Status mobile */}
                <div className="flex items-center gap-2 px-4 py-3 rounded-xl glass-card">
                  <Activity 
                    size={16} 
                    className={`${apiStatus?.available ? 'text-green-500' : 'text-gray-400'}`}
                  />
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    {apiStatus?.available ? 'API Connected' : 'API Offline'}
                  </span>
                </div>

                {/* Mobile links */}
                <button
                  onClick={() => {
                    setShowInfo(true)
                    setMobileMenuOpen(false)
                  }}
                  className="w-full flex items-center gap-3 px-4 py-3 rounded-xl glass-card text-gray-700 dark:text-gray-300 font-medium"
                >
                  <Info size={18} />
                  About
                </button>

                <a
                  href="https://github.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-full flex items-center gap-3 px-4 py-3 rounded-xl glass-card text-gray-700 dark:text-gray-300 font-medium"
                  onClick={() => setMobileMenuOpen(false)}
                >
                  <Github size={18} />
                  GitHub
                </a>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.header>

      {/* Info Modal */}
      <AnimatePresence>
        {showInfo && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setShowInfo(false)}
            className="fixed inset-0 z-[100] bg-black/60 backdrop-blur-sm flex items-center justify-center p-4"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="max-w-lg w-full glass-card p-8 shadow-2xl"
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-black gradient-text">About ReviewAuth</h2>
                <button
                  onClick={() => setShowInfo(false)}
                  className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                >
                  <X size={24} className="text-gray-600 dark:text-gray-400" />
                </button>
              </div>

              <div className="space-y-4 text-gray-600 dark:text-gray-400">
                <p>
                  <strong className="text-gray-900 dark:text-gray-100">ReviewAuth</strong> is an advanced AI-powered system designed to detect fake reviews on e-commerce platforms.
                </p>
                
                <div className="space-y-2">
                  <h3 className="font-bold text-gray-900 dark:text-gray-100">Features:</h3>
                  <ul className="list-disc list-inside space-y-1 text-sm">
                    <li>Multi-model ensemble predictions</li>
                    <li>SHAP-based explanations</li>
                    <li>Confidence visualization</li>
                    <li>Batch processing support</li>
                    <li>Real-time analysis</li>
                  </ul>
                </div>

                <div className="pt-4 border-t border-gray-200 dark:border-gray-800">
                  <p className="text-sm">
                    Built with React, Flask, scikit-learn, and powered by machine learning algorithms.
                  </p>
                </div>
              </div>

              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setShowInfo(false)}
                className="w-full mt-6 btn-primary"
              >
                Got it!
              </motion.button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}