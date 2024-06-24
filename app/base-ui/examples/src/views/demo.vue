<template>
  <ElAffix type="success">123</ElAffix>
  <EnumSelectLocal :data="ccc" placeholder="本地组件"></EnumSelectLocal>
  <ec-icon-select></ec-icon-select>
  <EnumSelect :data="ccc" placeholder="远程组件"></EnumSelect>
</template>
<script setup lang="ts">
import { ref } from 'vue'
//import { EcIconSelect } from 'co6co'
import { IEnumSelect, EnumSelect } from 'co6co'
import { EcIconSelect, EnumSelect as EnumSelectLocal } from '../components'
import { ElAffix } from 'element-plus'
import { get_state_svc } from '../api/boat/jobs'

const ccc = ref<IEnumSelect[]>([])
get_state_svc().then((res) => {
  ccc.value = res.data.stateslist
})
</script>
