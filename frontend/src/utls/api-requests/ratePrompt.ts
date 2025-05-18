import type { ModelTypes } from "@/types/possibleModelTypes";
import { ApiClient } from "@/utls/api-requests/apiclient";

export async function ratePrompt(prompt: string, model: ModelTypes){
  try{
  const response = await ApiClient.get('rate-prompt', {params: {prompt: prompt, model: model}})
  return response.data
  }
  catch{
    return undefined
  }
}
