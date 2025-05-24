<script lang="ts" setup>
import { ref } from 'vue'
// 👉 Replace with your actual request helper

// ───────────────────────── State ──────────────────────────
const email = ref('')
const username = ref('')
const password = ref('')
const confirmPassword = ref('')

const loading = ref(false)
const error = ref('')
const success = ref(false)

// ──────────────────────── Methods ─────────────────────────
async function handleRegister() {
  error.value = ''

  if (password.value !== confirmPassword.value) {
    error.value = 'Passwords do not match'
    return
  }

  try {
    loading.value = true
    success.value = true
  } catch (e: any) {
    error.value = e?.message ?? 'Registration failed'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="register-page">
    <h1 class="register-page__title">Create your account</h1>

    <form class="register-page__form" @submit.prevent="handleRegister">
      <!-- ──────── Email ──────── -->
      <label class="register-page__label">
        E‑mail
        <input
          class="register-page__input"
          type="email"
          autocomplete="email"
          v-model.trim="email"
          required
        />
      </label>

      <!-- ─────── Username ────── -->
      <label class="register-page__label">
        Username
        <input
          class="register-page__input"
          type="text"
          autocomplete="username"
          v-model.trim="username"
          required
        />
      </label>

      <!-- ─────── Password ────── -->
      <label class="register-page__label">
        Password
        <input
          class="register-page__input"
          type="password"
          autocomplete="new-password"
          v-model.trim="password"
          required
          minlength="6"
        />
      </label>

      <!-- ── Confirm password ─── -->
      <label class="register-page__label">
        Confirm password
        <input
          class="register-page__input"
          type="password"
          autocomplete="new-password"
          v-model.trim="confirmPassword"
          required
          minlength="6"
        />
      </label>

      <!-- ───────── Actions ──────── -->
      <button class="simple-button register-page__submit" :disabled="loading" type="submit">
        {{ loading ? 'Registering…' : 'Register' }}
      </button>

      <!-- ────────── Alert area ───────── -->
      <p v-if="error" class="register-page__alert register-page__alert--error">{{ error }}</p>
      <p v-if="success" class="register-page__alert register-page__alert--success">Account created! You can now log in.</p>
    </form>
  </div>
</template>

<style lang="scss" scoped>
.register-page {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  height: calc(100vh - 45px);
  width: 100vw;
  padding: 40px 20px;
  overflow-y: auto;

  &__title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 40px;
    color: $text-color;
  }

  &__form {
    width: 100%;
    max-width: 420px;
    display: grid;
    grid-template-columns: 1fr;
    gap: 20px;
  }

  &__label {
    display: grid;
    gap: 6px;
    font-weight: 600;
    color: $text-color;
  }

  &__input {
    padding: 12px 16px;
    font-size: 1rem;
    border-radius: 15px;
    border: 2px solid $main-color;
    background-color: $background-color;
    transition: background-color 150ms ease;

    &:focus {
      outline: none;
      background-color: $shadowed-background-color;
    }
  }

  &__submit {
    width: 100%;
    justify-self: center;
  }

  &__alert {
    text-align: center;
    padding: 8px 12px;
    border-radius: 12px;
    font-weight: 600;

    &--error {
      background-color: rgba(red($danger-color), green($danger-color), blue($danger-color), 0.15);
      color: $danger-color;
    }

    &--success {
      background-color: rgba(red($success-color), green($success-color), blue($success-color), 0.15);
      color: $success-color;
    }
  }
}
</style>
