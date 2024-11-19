<template>
  <div class="container-layout c-container">
    <el-container>
      <el-header
        ><span style="position: absolute; left: 0; top: 0"
          >{{ zoom }}--{{ BaiduMapAK }}</span
        ></el-header
      >
      <el-main @wheel="onWheel">
        <BaiduMap
          :center="center"
          :zoom="zoom"
          :ak="BaiduMapAK"
          @ready="handler"
          :scroll-wheel-zoom="false"
        >
          <!-- 在这里添加地图元素 -->
          <BmMarker :position="markerPosition" :draggable="true" lanel="中部">
            <template #label>
              <span>北京</span>
            </template>
          </BmMarker>
          <!--图层-->
          <BmMapType
            :map-types="['BMAP_NORMAL_MAP', 'BMAP_SATELLITE_MAP', 'BMAP_TRAFFIC_MAP']"
          ></BmMapType>
        </BaiduMap>
      </el-main>
    </el-container>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onBeforeMount, ref } from 'vue'
// 安装 vue-baidu-map-3x
// npm install vue-baidu-map-3x
import { BaiduMap, BmMarker, BmMapType } from 'vue-baidu-map-3x'
import { ElMain, ElContainer } from 'element-plus'
import { Point, showLoading, closeLoading } from 'co6co'
import { configSvc } from 'co6co-right'
import { ConfigCodes } from '../api/app'
const center = ref<Point>({ lng: 102.927641, lat: 25.095627 })
const zoom = ref(12)

const BaiduMapAK = ref<string>()
const featData = async () => {
  console.info('loading..')
  const res = await configSvc.get_config_svc(ConfigCodes.BaiduMapAK)
  console.info('loading..', res)
  //BaiduMapAK.value = res.data.value
}
onMounted(() => {
  featData()
})
const handler = ({ BMap, map }) => {
  // 监听地图点击事件
  map.addEventListener('click', function (e) {
    console.log('Clicked at:', e.point.lng, e.point.lat)
  })
}
const onWheel = (event) => {
  if (event.deltaY < 0) {
    // 滚轮向上，放大地图
    zoom.value = Math.min(zoom.value + 1, 19)
  } else if (event.deltaY > 0) {
    // 滚轮向下，缩小地图
    zoom.value = Math.max(zoom.value - 1, 1)
  }
  console.info(zoom.value)
}
const markerPoint = { ...center.value }
const markerPosition = ref<{ lng: number; lat: number }>(markerPoint)
</script>

<style scoped lang="less">
.el-main {
  width: 100%;
  height: 100%;
  padding: 0;
  & > div:first-child {
    width: 100%;
    height: 100%;
  }
}
</style>
