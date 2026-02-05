// import { useEffect, useState } from "react";
// import { motion, AnimatePresence } from "framer-motion";
// import { Sun, Moon } from "lucide-react";

// export default function DarkModeToggle() {
//   const [dark, setDark] = useState(() =>
//     window.matchMedia("(prefers-color-scheme: dark)").matches
//   );

//   useEffect(() => {
//     const root = document.documentElement;
//     if (dark) {
//       root.classList.add("dark");
//     } else {
//       root.classList.remove("dark");
//     }
//   }, [dark]);

//   return (
//     <motion.button
//       onClick={() => setDark((d) => !d)}
//       whileHover={{ scale: 1.1, rotateY: 10 }}
//       whileTap={{ scale: 0.95 }}
//       transition={{ type: "spring", stiffness: 400, damping: 17 }}
//       className="relative group flex items-center justify-center w-14 h-14 rounded-2xl 
//                  backdrop-blur-xl bg-gray-800/40 border border-gray-600/30
//                  shadow-2xl hover:shadow-amber-500/20
//                  hover:border-amber-500/50 focus:outline-none focus:ring-2 focus:ring-amber-500/50
//                  transition-all duration-500"
//       aria-label="Toggle theme"
//       title="Toggle theme"
//     >
//       {/* Premium Glow Effect */}
//       <motion.div
//         className="absolute inset-0 rounded-2xl bg-gradient-to-r from-amber-500/20 to-orange-500/20 opacity-0 group-hover:opacity-100 blur-sm transition-opacity duration-500"
//         layoutId="glow"
//       />
      
//       {/* Corner Accents */}
//       <div className="absolute top-1 left-1 w-3 h-3 border-l-2 border-t-2 border-amber-500/60 rounded-tl-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
//       <div className="absolute top-1 right-1 w-3 h-3 border-r-2 border-t-2 border-amber-500/60 rounded-tr-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
//       <div className="absolute bottom-1 left-1 w-3 h-3 border-l-2 border-b-2 border-amber-500/60 rounded-bl-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
//       <div className="absolute bottom-1 right-1 w-3 h-3 border-r-2 border-b-2 border-amber-500/60 rounded-br-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>

//       {/* Icon Container */}
//       <div className="relative z-10">
//         <AnimatePresence mode="wait" initial={false}>
//           {dark ? (
//             <motion.span
//               key="moon"
//               initial={{ rotate: -180, opacity: 0, scale: 0.3 }}
//               animate={{ rotate: 0, opacity: 1, scale: 1 }}
//               exit={{ rotate: 180, opacity: 0, scale: 0.3 }}
//               transition={{ 
//                 duration: 0.6,
//                 type: "spring",
//                 stiffness: 200,
//                 damping: 20
//               }}
//               className="text-amber-400 drop-shadow-lg"
//             >
//               <motion.div
//                 animate={{ 
//                   filter: [
//                     "drop-shadow(0 0 0px rgb(251 191 36))",
//                     "drop-shadow(0 0 8px rgb(251 191 36))",
//                     "drop-shadow(0 0 0px rgb(251 191 36))"
//                   ]
//                 }}
//                 transition={{ duration: 2, repeat: Infinity }}
//               >
//                 <Moon size={24} strokeWidth={2.5} />
//               </motion.div>
//             </motion.span>
//           ) : (
//             <motion.span
//               key="sun"
//               initial={{ rotate: 180, opacity: 0, scale: 0.3 }}
//               animate={{ rotate: 0, opacity: 1, scale: 1 }}
//               exit={{ rotate: -180, opacity: 0, scale: 0.3 }}
//               transition={{ 
//                 duration: 0.6,
//                 type: "spring",
//                 stiffness: 200,
//                 damping: 20
//               }}
//               className="text-orange-400 drop-shadow-lg"
//             >
//               <motion.div
//                 animate={{ 
//                   rotate: [0, 360],
//                   filter: [
//                     "drop-shadow(0 0 0px rgb(251 146 60))",
//                     "drop-shadow(0 0 12px rgb(251 146 60))",
//                     "drop-shadow(0 0 0px rgb(251 146 60))"
//                   ]
//                 }}
//                 transition={{ 
//                   rotate: { duration: 8, repeat: Infinity, ease: "linear" },
//                   filter: { duration: 2, repeat: Infinity }
//                 }}
//               >
//                 <Sun size={24} strokeWidth={2.5} />
//               </motion.div>
//             </motion.span>
//           )}
//         </AnimatePresence>
//       </div>

//       {/* Premium Status Indicator */}
//       <motion.div
//         className={`absolute -bottom-1 -right-1 w-4 h-4 rounded-full border-2 border-gray-900 ${
//           dark 
//             ? "bg-gradient-to-r from-amber-400 to-yellow-500" 
//             : "bg-gradient-to-r from-orange-400 to-red-500"
//         }`}
//         initial={{ scale: 0 }}
//         animate={{ scale: 1 }}
//         transition={{ delay: 0.2, type: "spring", stiffness: 500, damping: 30 }}
//       >
//         <motion.div
//           className="absolute inset-0 rounded-full bg-current opacity-50 blur-sm"
//           animate={{ scale: [1, 1.5, 1] }}
//           transition={{ duration: 2, repeat: Infinity }}
//         />
//       </motion.div>

//       {/* Ripple Effect on Click */}
//       <motion.div
//         className="absolute inset-0 rounded-2xl border-2 border-amber-500 opacity-0"
//         animate={{ scale: [1, 1.2], opacity: [0.5, 0] }}
//         transition={{ duration: 0.6 }}
//         key={`ripple-${dark}`}
//       />
//     </motion.button>
//   );
// }












import { useEffect, useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Sun, Moon } from "lucide-react"

export default function DarkModeToggle() {
  const [dark, setDark] = useState(() => {
    // Check localStorage first, then system preference
    const stored = localStorage.getItem('darkMode')
    if (stored !== null) {
      return stored === 'true'
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches
  })

  useEffect(() => {
    const root = document.documentElement
    if (dark) {
      root.classList.add("dark")
    } else {
      root.classList.remove("dark")
    }
    // Save preference to localStorage
    localStorage.setItem('darkMode', dark.toString())
  }, [dark])

  return (
    <motion.button
      onClick={() => setDark((d) => !d)}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      className="relative group flex items-center justify-center w-12 h-12 rounded-xl 
                 bg-gray-100 dark:bg-gray-800 
                 hover:bg-gray-200 dark:hover:bg-gray-700
                 border border-gray-200 dark:border-gray-700
                 shadow-md hover:shadow-lg
                 transition-all duration-300
                 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
      aria-label="Toggle theme"
      title={dark ? "Switch to light mode" : "Switch to dark mode"}
    >
      {/* Animated background gradient */}
      <motion.div
        className="absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"
        style={{
          background: dark 
            ? 'linear-gradient(135deg, rgba(147, 51, 234, 0.1), rgba(236, 72, 153, 0.1))'
            : 'linear-gradient(135deg, rgba(251, 146, 60, 0.1), rgba(251, 191, 36, 0.1))'
        }}
      />

      {/* Icon Container with smooth transition */}
      <div className="relative z-10">
        <AnimatePresence mode="wait" initial={false}>
          {dark ? (
            <motion.div
              key="moon"
              initial={{ rotate: -90, opacity: 0, scale: 0.5 }}
              animate={{ rotate: 0, opacity: 1, scale: 1 }}
              exit={{ rotate: 90, opacity: 0, scale: 0.5 }}
              transition={{ 
                duration: 0.4,
                type: "spring",
                stiffness: 200,
                damping: 15
              }}
              className="flex items-center justify-center"
            >
              <motion.div
                animate={{ 
                  rotate: [0, -10, 10, -10, 0],
                }}
                transition={{ 
                  duration: 3,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
              >
                <Moon 
                  size={20} 
                  strokeWidth={2.5} 
                  className="text-purple-600 dark:text-purple-400 drop-shadow-lg"
                />
              </motion.div>
            </motion.div>
          ) : (
            <motion.div
              key="sun"
              initial={{ rotate: 90, opacity: 0, scale: 0.5 }}
              animate={{ rotate: 0, opacity: 1, scale: 1 }}
              exit={{ rotate: -90, opacity: 0, scale: 0.5 }}
              transition={{ 
                duration: 0.4,
                type: "spring",
                stiffness: 200,
                damping: 15
              }}
              className="flex items-center justify-center"
            >
              <motion.div
                animate={{ 
                  rotate: [0, 360],
                }}
                transition={{ 
                  duration: 20,
                  repeat: Infinity,
                  ease: "linear"
                }}
              >
                <Sun 
                  size={20} 
                  strokeWidth={2.5} 
                  className="text-orange-500 drop-shadow-lg"
                />
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Subtle glow effect on hover */}
      <motion.div
        className={`absolute inset-0 rounded-xl blur-md opacity-0 group-hover:opacity-30 transition-opacity duration-300 ${
          dark 
            ? 'bg-purple-500' 
            : 'bg-orange-400'
        }`}
      />

      {/* Click ripple effect */}
      <AnimatePresence>
        {/* This will be triggered on each click via key change */}
        <motion.div
          key={`ripple-${dark}`}
          className={`absolute inset-0 rounded-xl ${
            dark 
              ? 'border-2 border-purple-400' 
              : 'border-2 border-orange-400'
          }`}
          initial={{ scale: 1, opacity: 0.6 }}
          animate={{ scale: 1.5, opacity: 0 }}
          transition={{ duration: 0.5 }}
        />
      </AnimatePresence>
    </motion.button>
  )
}