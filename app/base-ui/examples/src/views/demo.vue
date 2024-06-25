<template>
  <EnumSelect :data="ccc" placeholder="远程组件" />
  <IconSelect></IconSelect>
</template>
<script setup lang="ts">
import { ref } from 'vue'
import { IEnumSelect } from 'co6co'
import { EnumSelect, IconSelect } from 'co6co'
//import { EcIconSelect, EnumSelect as EnumSelectLocal } from '../components'
import { get_state_svc } from '../api/boat/jobs'

const ccc = ref<IEnumSelect[]>([])
get_state_svc().then((res) => {
  ccc.value = res.data.stateslist
})
</script>
