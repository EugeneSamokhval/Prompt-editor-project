import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component:() =>  import("@/pages/home/HomePage.vue"),
    },
    {
      path: '/registration',
      name: 'register',
      component: () => import('@/pages/registration/RegistrationPage.vue'),
    },
  ],
})

export default router
