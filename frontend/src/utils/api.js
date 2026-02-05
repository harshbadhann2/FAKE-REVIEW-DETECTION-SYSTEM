// import axios from 'axios'

// const api = axios.create({
//   baseURL: import.meta.env.VITE_API_BASE || 'http://127.0.0.1:5000',
//   headers: { 'Content-Type': 'application/json' },
//   timeout: 15000  // Increased timeout for SHAP calculations
// })

// // Add request interceptor for debugging
// api.interceptors.request.use(request => {
//   console.log('🚀 API Request:', {
//     method: request.method.toUpperCase(),
//     url: request.url,
//     data: request.data
//   })
//   return request
// })

// // Add response interceptor for debugging
// api.interceptors.response.use(
//   response => {
//     console.log('✅ API Response:', {
//       status: response.status,
//       url: response.config.url,
//       data: response.data
//     })
//     return response
//   },
//   error => {
//     console.error('❌ API Error:', {
//       status: error.response?.status,
//       url: error.config?.url,
//       message: error.response?.data || error.message
//     })
//     return Promise.reject(error)
//   }
// )

// // Health check function
// export const healthCheck = async () => {
//   try {
//     const response = await api.get('/health')
//     return response.data
//   } catch (error) {
//     console.error('Health check failed:', error)
//     throw error
//   }
// }

// // Main prediction function
// export const predictReview = async (review) => {
//   try {
//     const response = await api.post('/predict', { review })
//     console.log('🎯 Prediction result:', response.data)
//     return response.data
//   } catch (error) {
//     console.error('Predict API error:', error)
//     throw new Error(
//       error.response?.data?.error || 
//       'Failed to analyze review. Please try again.'
//     )
//   }
// }

// // Explanation function
// export const explainReview = async (review) => {
//   try {
//     console.log('🔍 Requesting explanation for:', review.substring(0, 50) + '...')
//     const response = await api.post('/explain', { review })
//     console.log('🧠 Explanation result:', response.data)
//     return response.data
//   } catch (error) {
//     console.error('Explain API error:', error.response?.data || error.message)
    
//     // More specific error handling
//     if (error.response?.status === 400) {
//       throw new Error('Invalid review text provided')
//     } else if (error.response?.status === 500) {
//       throw new Error('Server error during explanation generation')
//     } else {
//       throw new Error('Failed to generate explanation. Please try again.')
//     }
//   }
// }

// export default api













import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000',
  headers: { 'Content-Type': 'application/json' },
  timeout: 20000  // Increased timeout for complex operations
})

// Add request interceptor for debugging
api.interceptors.request.use(request => {
  console.log('🚀 API Request:', {
    method: request.method.toUpperCase(),
    url: request.url,
    data: request.data
  })
  return request
})

// Add response interceptor for debugging and error handling
api.interceptors.response.use(
  response => {
    console.log('✅ API Response:', {
      status: response.status,
      url: response.config.url,
      data: response.data
    })
    return response
  },
  error => {
    console.error('❌ API Error:', {
      status: error.response?.status,
      url: error.config?.url,
      message: error.response?.data || error.message
    })
    return Promise.reject(error)
  }
)

// Health check function
export const healthCheck = async () => {
  try {
    const response = await api.get('/health')
    return response.data
  } catch (error) {
    console.error('Health check failed:', error)
    throw error
  }
}

// Get model information
export const getModelInfo = async () => {
  try {
    const response = await api.get('/model-info')
    return response.data
  } catch (error) {
    console.error('Model info failed:', error)
    throw new Error('Failed to fetch model information')
  }
}

// Main prediction function with enhanced response
export const predictReview = async (review) => {
  try {
    const response = await api.post('/predict', { review })
    console.log('🎯 Prediction result:', response.data)
    return response.data
  } catch (error) {
    console.error('Predict API error:', error)
    throw new Error(
      error.response?.data?.error || 
      'Failed to analyze review. Please try again.'
    )
  }
}

// Explanation function with SHAP values
export const explainReview = async (review) => {
  try {
    console.log('🔍 Requesting explanation for:', review.substring(0, 50) + '...')
    const response = await api.post('/explain', { review })
    console.log('🧠 Explanation result:', response.data)
    return response.data
  } catch (error) {
    console.error('Explain API error:', error.response?.data || error.message)
    
    // Specific error handling
    if (error.response?.status === 400) {
      throw new Error(error.response.data?.error || 'Invalid review text provided')
    } else if (error.response?.status === 500) {
      throw new Error('Server error during explanation generation')
    } else {
      throw new Error('Failed to generate explanation. Please try again.')
    }
  }
}

// Visualization function - returns image data
export const visualizeReview = async (review) => {
  try {
    console.log('📊 Requesting visualization for review')
    const response = await api.post('/visualize', { review })
    console.log('📈 Visualization generated')
    return response.data
  } catch (error) {
    console.error('Visualize API error:', error)
    throw new Error(
      error.response?.data?.error || 
      'Failed to generate visualization'
    )
  }
}

// Batch prediction function
export const batchPredict = async (reviews) => {
  try {
    if (!Array.isArray(reviews) || reviews.length === 0) {
      throw new Error('Please provide an array of reviews')
    }
    
    if (reviews.length > 100) {
      throw new Error('Maximum 100 reviews per batch')
    }
    
    console.log(`📦 Batch predicting ${reviews.length} reviews`)
    const response = await api.post('/batch-predict', { reviews })
    console.log('✅ Batch prediction complete:', response.data.summary)
    return response.data
  } catch (error) {
    console.error('Batch predict API error:', error)
    throw new Error(
      error.response?.data?.error || 
      'Failed to process batch prediction'
    )
  }
}

// Combined analysis - get prediction + explanation + visualization
export const fullAnalysis = async (review) => {
  try {
    console.log('🔬 Starting full analysis')
    
    // Run prediction and explanation in parallel
    const [predictionData, explanationData] = await Promise.all([
      predictReview(review),
      explainReview(review).catch(err => {
        console.warn('Explanation failed, continuing without it:', err.message)
        return null
      })
    ])
    
    // Get visualization (optional)
    let visualizationData = null
    try {
      visualizationData = await visualizeReview(review)
    } catch (err) {
      console.warn('Visualization failed, continuing without it:', err.message)
    }
    
    return {
      prediction: predictionData,
      explanation: explanationData,
      visualization: visualizationData
    }
  } catch (error) {
    console.error('Full analysis error:', error)
    throw error
  }
}

// Utility function to check API availability
export const checkApiAvailability = async () => {
  try {
    const health = await healthCheck()
    return {
      available: health.status === 'healthy',
      status: health.status,
      details: health
    }
  } catch (error) {
    return {
      available: false,
      status: 'unavailable',
      error: error.message
    }
  }
}

export default api