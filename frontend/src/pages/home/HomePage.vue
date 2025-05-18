<script lang="ts" setup>
import { ref } from 'vue'
import { tokenizePrompt } from '@/utls/api-requests/tokenizePrompt'
import type { PromptStructure } from '@/types/promptStructure.d'
import { ModelTypes } from '@/types/possibleModelTypes.d'
import { testPrompt } from '@/utls/api-requests/testPrompt'
import { ratePrompt } from '@/utls/api-requests/ratePrompt'
const textPreview = ref('')
const imagePreview = ref<string>('')
const inputText = ref('')
const ratingText = ref(0)
const model = ref<ModelTypes>(ModelTypes.FALLBACK)
const allModelTypes = ref([ModelTypes.CHAT_GPT, ModelTypes.FALLBACK, ModelTypes.STABLE_DIFFUSION])
const tokenized = ref<PromptStructure | null>()
type SectionKey = keyof PromptStructure //  convenience alias

const handleTokenization = async () => {
  const data = await tokenizePrompt(inputText.value)
  tokenized.value = data
}
const handleTest = async () => {
  const response_preview = await testPrompt(inputText.value, model.value)
  console.log(response_preview)
  imagePreview.value = response_preview?.image || ''
  textPreview.value = response_preview?.text || ''
}

const handleRating = async () => {
  const rating = await ratePrompt(inputText.value, model.value)
  ratingText.value = rating
}
// Drag an drop code
function onDragStart(e: DragEvent, section: SectionKey, index: number) {
  console.log('Dragging')
  if (e.dataTransfer) {
    e.dataTransfer.setData('text/plain', `${section}|${index}`)
    e.dataTransfer.effectAllowed = 'move'
  }
}

/** Allow the item we’re hovering over to accept a drop */
function onDragOver(e: DragEvent) {
  console.log('Dragging over')
  e.preventDefault() // ← **critical**: makes it droppable
  e.dataTransfer!.dropEffect = 'move'
}

/** Re-insert the dragged phrase at its new position */
function onDrop(e: DragEvent, section: SectionKey, dropIndex: number) {
  console.log('Dropping')
  e.preventDefault()

  // Pull the origin we stashed in onDragStart()
  const [dragSection, dragIdxStr] = (e.dataTransfer?.getData('text/plain') || '').split('|')
  const dragIndex = Number(dragIdxStr)

  // Clone the list → splice out dragged item → splice it back in at dropIndex
  const current = tokenized.value?.[section]
  if (!current) return

  const reordered = [...current]
  const [moved] = reordered.splice(dragIndex, 1)
  reordered.splice(dropIndex, 0, moved)

  // Tell Vue the section changed (spread keeps reactivity)
  tokenized.value = { ...tokenized.value!, [section]: reordered }
}

//Combine into a new prompt
function connectPrompt() {
  inputText.value = tokenized.value
    ? tokenized.value?.clarity.join(', ') +
      tokenized.value?.composition.join(', ') +
      tokenized.value?.context.join(', ') +
      tokenized.value?.descriptive.join(', ') +
      tokenized.value?.lighting.join(', ') +
      tokenized.value?.negative.join(', ') +
      tokenized.value?.style.join(', ') +
      tokenized.value?.technical.join(', ')
    : inputText.value
}
</script>

<template>
  <div class="home-page">
    <div>
      <textarea class="home-page__text-input" v-model="inputText"></textarea>
      <div class="home-page__buttons-box">
        <select class="simple-button" v-model="model">
          <option v-for="model in allModelTypes">{{ model }}</option>
        </select>
        <button class="simple-button" @click="handleTokenization">Tokenize prompt</button>
        <button class="simple-button" @click="handleTest">Test prompt</button>
        <button class="simple-button" @click="handleRating">Rate prompt</button>
      </div>
      <div v-if="tokenized" class="home-page__result-output-field">
        <template v-for="(list, section) in tokenized" :key="section">
          <div class="result-output-field__type-key">{{ section }}:</div>
          <button
            v-for="(phrase, i) in list"
            :key="`${section}-${i}`"
            class="result-output-field__text-input-item"
            :draggable="true"
            @dragstart="onDragStart($event, section as SectionKey, i)"
            @dragover="onDragOver"
            @drop="onDrop($event, section as SectionKey, i)"
          >
            {{ phrase }}
          </button>
        </template>
      </div>
      <button @click="connectPrompt" class="simple-button">Save to prompt</button>
    </div>
    <div class="home-page__output-box">
      <p class="output-box__prompt-rating" :ratingText>Current prompt rating:{{ ratingText }}</p>
      <img :imagePreview alt="Generated image" v-if="imagePreview.length > 0" :src="imagePreview" />
      <p :textPreview v-if="textPreview.length > 0" v-text="textPreview"></p>
    </div>
  </div>
</template>

<style lang="scss">
.home-page {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 60px;
  gap: 20px;
 height: calc(100vh - 45px);
  width: 100vw;
  padding: 20px;
}
.home-page__text-input {
  height: 500px;
  width: 900px;
  font-weight: 500;
  font-size: 18px;
  border-radius: 15px;
}
.home-page__result-output-field {
  display: grid;
  grid-template-columns: repeat(auto-fill, 120px);
  grid-template-rows: repeat(auto-fill, 30px);
  grid-column: span 2;
}
.result-output-field__text-input-item {
  border-radius: 15px;
  border: 2px solid $main-color;
  display: grid;
  justify-content: center;
  background-color: $background-color;
  cursor: pointer;
  user-select: none;
}
.result-output-field__text-input-item.dragging {
  opacity: 0.4;
}
.result-output-field__text-input-item:hover {
  background-color: $shadowed-background-color;
}
.home-page__buttons-box {
  display: grid;
  grid-template-rows: 1fr 1fr 1fr 1fr;
  grid-template-columns: 1fr;
  gap: 10px;
  grid-column: span 2;
  padding-top: 10px;
}
.result-output-field__type-key {
  border: 1px solid $secondary-color;
  border-radius: 15px;
  align-content: center;
  text-align: center;
}
.home-page__output-box {
  display: grid;
  grid-template-columns: 1fr;
  max-width: 900px;
  max-height: 100%;
  overflow-y: auto;
  border-left: 2px solid $text-color;
}
.output-box__prompt-rating {
  width: 200px;
  height: 20px;
  font-weight: 700;
  border-radius: 1234px;
  padding: 4px;
}
</style>
