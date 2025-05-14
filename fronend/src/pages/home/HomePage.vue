<script lang="ts" setup>
import { ref } from 'vue'
import { tokenizePrompt } from '@/utls/api-requests/tokenizePrompt'
import type { PromptStructure } from '@/types/promptStructure.d'

const inputText = ref('')

const tokenized = ref<PromptStructure | null>()
const handleTokenization = async () => {
  const data = await tokenizePrompt(inputText.value)
  console.log(data)
  tokenized.value = data
}
</script>

<template>
  <div class="home-page">
  <textarea class="home-page__text-input" v-model="inputText"></textarea>

  <button @click="handleTokenization">Test prompt</button>

  <div v-if="tokenized">
    <template v-for="(list, section) in tokenized" :key="section">
      <div v-for="(phrase, i) in list" :key="`${section}-${i}`" class="text-input-item">
        {{ phrase }}
      </div>
    </template>
  </div>
  </div>
</template>

<style lang="scss">
.home-page{
  display: grid;
}
.home-page__text-input{
  height: 200px;
  width: 1000px;
  font-weight: 500;
  font-size: 18px;
  border-radius: 15px;
}
</style>

