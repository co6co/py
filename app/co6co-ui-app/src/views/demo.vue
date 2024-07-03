<template>
  <hello :count="123" :person="{ name: '正' }" color="error"></hello>
  <EnumSelect :data="ccc" placeholder="远程组件" />
  <IconSelect></IconSelect>
</template>
<script setup lang="ts">
import { ref } from 'vue'
import { EnumSelect, IEnumSelect, IconSelect, Hello } from 'co6co'
//import { EcIconSelect, EnumSelect as EnumSelectLocal } from '../components'
import { get_state_svc } from '../api/boat/jobs'

const ccc = ref<IEnumSelect[]>([])
get_state_svc().then((res) => {
  ccc.value = res.data.stateslist
})
</script>
