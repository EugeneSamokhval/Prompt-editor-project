<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
</script>

<template>
  <header>
    <div class="wrapper">
      <nav class="wrapper__nav-buttons-layout">
        <RouterLink class="nav-buttons-layout__nav-button" :class="{'active':$route.name === 'home'}" to="/">Home</RouterLink>
        <RouterLink class="nav-buttons-layout__nav-button" :class="{'active':$route.name === 'register'}" to="/registration">Registration</RouterLink>
      </nav>
    </div>
  </header>
  <main class="main-content">
  <RouterView/>
  </main>
</template>

<style lang="scss" scoped>
.wrapper{
  display: grid;
  grid-template-rows: 1fr;
  grid-template-columns: 1fr;
}
.wrapper__nav-buttons-layout{
  display: grid;
  grid-template-columns: 200px 200px;
  grid-template-rows: 1fr;
  gap: 10px;
}
.nav-buttons-layout__nav-button{
  background-color: $secondary-color;
  color: $background-color;
}
.active{
  background-color: $main-color;
}
.main-content{
  display: grid;
  grid-template-columns: 1fr;
  grid-template-rows: 1fr;
  align-items: center;
  justify-items: center;
  width: 100%;
  height: 100vh;

}
</style>
