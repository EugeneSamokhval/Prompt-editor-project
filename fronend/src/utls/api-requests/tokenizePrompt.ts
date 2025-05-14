import type { PromptStructure } from "@/types/promptStructure.d";
import { ApiClient } from "@/utls/api-requests/apiclient";

export async function tokenizePrompt(prompt: string){
  try{
  const response = await ApiClient.get('/tokenize-prompt', {params: {prompt: prompt}})
  return response.data as PromptStructure
  }
  catch{
    return undefined
  }
}
