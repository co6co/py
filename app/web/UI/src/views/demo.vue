<template>
  <div class="error-page">
    <span v-non-permiss="getPermissKey(routeHook.ViewFeature.view)"> 没有权限</span>
    <span v-permiss="getPermissKey(routeHook.ViewFeature.view)"> 有权限 {{ Data }}</span>
    <GaodePoint v-model="Data" style="height: 80vh; width: 90%" />
  </div>
</template>

<script setup lang="ts">
import { routeHook } from 'co6co-right'
import { ref } from 'vue'
import GaodePoint from '../components/location/gaodePoint'
const { getPermissKey, getCurrentViewFeature } = routeHook.usePermission()
console.info(getPermissKey(routeHook.ViewFeature.view))
console.info(getCurrentViewFeature())

const { getRouteData } = routeHook.useRouteData()
const data = getRouteData()
console.info(data)
const Data = ref({ lng: 102.927641, lat: 25.095627 })
</script>
