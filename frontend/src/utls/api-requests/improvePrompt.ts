import { ApiClient } from "@/utls/api-requests/apiclient";

export async function improvePrompt(prompt: string){
  try{
  const response = await ApiClient.get('improve-prompt', {params: {prompt: prompt}})
  return response.data
  }
  catch{
    return undefined
  }
}
