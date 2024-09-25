<template>
  <div class="container-layout">
    <el-container>
      <el-main>
        <div id="container"></div>
      </el-main>
    </el-container>
  </div>
</template>
<script setup>
// 安装 amap-jsapi-loader
// npm install @amap/amap-jsapi-loader

import { ref, onMounted } from 'vue'
import { load, AMap } from '@amap/amap-jsapi-loader'
import { showLoading, closeLoading } from 'co6co'

import * as icon from '@element-plus/icons-vue'
import { configSvc } from 'co6co-right'
import { ConfigCodes } from '../api/app'

const map = ref(null)
const center = [102.927641, 25.095627]
const fetchData = async () => {
  showLoading()
  const res = await configSvc.get_config_svc(ConfigCodes.GaoDeMapAK)
  closeLoading()
  return res
}

onMounted(async () => {
  const res = await fetchData()
  const AMap = await load({
    key: res.data.value, //'1a309adbc6e1d16cbf2a6b5970a95a79',
    version: '2.0',
    plugins: ['AMap.MapType'],
    opacity: 0.8
  })

  map.value = new AMap.Map('container', {
    center: center,
    zoom: 11,
    mapStyle: 'amap://styles/normal',
    plugins: ['AMap.MapType']
  })

  // 监听地图点击事件
  map.value.on('click', function (e) {
    console.log('Clicked at:', e.lnglat)

    const marker = new AMap.Marker({
      position: e.lnglat, // 使用点击事件中的经纬度
      title: '点击创建的标注' + e.lnglat,
      //cursor: 'pointer',
      //icon: icon.AddLocation,
      shadow: true
      //content: e.lnglat
    })
    marker.setMap(map.value)
  })

  // 创建标注
  const marker = new AMap.Marker({
    position: center, // 标注点的经纬度
    title: '中心点标注'
  })
  map.value.add(marker)
})
</script>

<style scoped lang="less">
.el-main {
  width: 100%;
  height: 100%;
  padding: 0;
  #container {
    width: 100%;
    height: 100%;
  }
}
</style>
