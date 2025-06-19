<script lang="ts" setup>
import { ref } from 'vue'
// 👉 Swap in your real auth helper

// ───────────────────────── State ──────────────────────────
const identifier = ref('')   // E-mail or username
const password   = ref('')

const loading = ref(false)
const error   = ref('')
const success = ref(false)

// ──────────────────────── Methods ─────────────────────────
async function handleLogin() {
  error.value = ''
  success.value = false

  try {
    loading.value = true
    /* await auth.login({ identifier: identifier.value, password: password.value }) */
    success.value = true
  } catch (e: any) {
    error.value = e?.message ?? 'Login failed'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-page">
    <h1 class="login-page__title">Welcome back</h1>

    <form class="login-page__form" @submit.prevent="handleLogin">
      <!-- ─────── Identifier ─────── -->
      <label class="login-page__label">
        E-mail or username
        <input
          class="login-page__input"
          type="text"
          autocomplete="username"
          v-model.trim="identifier"
          required
        />
      </label>

      <!-- ─────── Password ──────── -->
      <label class="login-page__label">
        Password
        <input
          class="login-page__input"
          type="password"
          autocomplete="current-password"
          v-model.trim="password"
          required
          minlength="6"
        />
      </label>

      <!-- ───────── Actions ──────── -->
      <button class="simple-button login-page__submit" :disabled="loading" type="submit">
        {{ loading ? 'Logging in…' : 'Log in' }}
      </button>

      <!-- ────────── Alert area ───────── -->
      <p v-if="error"   class="login-page__alert login-page__alert--error">{{ error }}</p>
      <p v-if="success" class="login-page__alert login-page__alert--success">Signed in! Redirecting…</p>
    </form>
  </div>
</template>

<style lang="scss" scoped>
.login-page {
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
