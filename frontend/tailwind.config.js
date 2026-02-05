/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        amazon: {
          orange: '#FF9900',
          blue: '#232F3E',
          yellow: '#FEBD69'
        }
      },
      boxShadow: {
        header: '0 2px 12px rgba(0,0,0,0.08)'
      }
    }
  },
  plugins: []
}