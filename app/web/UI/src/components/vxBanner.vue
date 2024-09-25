<template>
  <div class="block text-center" m="t-4" style="overflow: hidden">
    <el-carousel trigger="touchstart">
      <el-carousel-item v-for="item in 4" :key="item">
        <h3
          class="small justify-center"
          text="2xl"
          :style="{ 'background-image': `url(${getBannerUrl(item)})` }"
        >
          {{ item }}
        </h3>
      </el-carousel-item>
    </el-carousel>
  </div>
</template>
<script setup>
import { getPublicURL } from '../utils'
import { randomString } from 'co6co'
const random = randomString(10, '1234567890')
const getBannerUrl = (index) => {
  return getPublicURL(`/banner/${index}.png?t=${random}`)
}
</script>
<style scoped>
.el-carousel {
  height: 150px !important;
}
.el-carousel__item h3 {
  color: #475669;
  line-height: 150px;
  margin: 0;
  text-align: center;

  background-repeat: no-repeat; /* 图像不重复 */
  background-position: center center; /* 图像居中显示 */
  background-size: cover; /* 图像将伸展或缩小以覆盖整个元素 */
}

.el-carousel__item:nth-child(2n) {
  background-color: #99a9bf;
}

.el-carousel__item:nth-child(2n + 1) {
  background-color: #d3dce6;
}
</style>
