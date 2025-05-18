import { ModelTypes } from '@/types/possibleModelTypes.d'
import { ApiClient } from '@/utls/api-requests/apiclient'

export async function testPrompt(prompt: string, model: ModelTypes) {
  try {
    if (model === ModelTypes.CHAT_GPT) {
      const response = await ApiClient.get('preview-text', { params: { prompt: prompt } })
      return {text: response.data, image: null}
    } else if (model === ModelTypes.STABLE_DIFFUSION) {
      const imageResponse = await ApiClient.get('preview-image', { params: { prompt: prompt },  responseType: 'blob' })
      const imageUrl = URL.createObjectURL(imageResponse.data)
      return {image: imageUrl, text: null}
    } else {
      const response = await ApiClient.get('preview-text', { params: { prompt: prompt } })
      const imageResponse = await ApiClient.get('preview-image', { params: { prompt: prompt },  responseType: 'blob'  })
      const imageURL = URL.createObjectURL(imageResponse.data)
      return { image: imageURL, text: response.data }
    }
  } catch {
    return undefined
  }
}
