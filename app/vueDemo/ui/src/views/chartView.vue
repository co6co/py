<template>
  <div>
    <a href="https://juejin.cn/post/7337159704956780594">参考</a>
    <div class="box"> 
        <div >
          <v-chart class="chart" :option="option" />
          <button @click="updateChartData" class="update-chart-btn">
            更新数据
          </button>
        </div>
        <div  >
          <v-chart class="chart" :option="barOption" />
          <button @click="updateBarData" class="update-chart-btn">
            更新柱状图数据
          </button>
        </div> 
        
    </div>
  </div>
</template>

   

<script setup>
import { ref } from 'vue'
import VChart from 'vue-echarts'
//import { use, graphic } from 'echarts/core'
//import { CanvasRenderer } from 'echarts/renderers'
//import { BarChart } from 'echarts/charts'
//import { TitleComponent, TooltipComponent, LegendComponent } from 'echarts/components'
//
//use([CanvasRenderer, BarChart, TitleComponent, TooltipComponent, LegendComponent])
 

 // 你不需要在组件中直接引入 echarts，因为 vue-echarts 已经处理了初始化
 import * as echarts from 'echarts';
 
 // 定义图表配置作为响应式数据
 const option = ref({
   title: {
     text: 'Referer of a Website',
     subtext: 'Fake Data',
     left: 'center'
   },
   tooltip: {
     trigger: 'item'
   },
   legend: {
     orient: 'vertical',
     left: 'left'
   },
   series: [
     {
       name: 'Access From',
       type: 'pie',
       radius: '50%',
       data: [
         { value: 1048, name: 'Search Engine' },
         { value: 735, name: 'Direct' },
         { value: 580, name: 'Email' },
         { value: 484, name: 'Union Ads' },
         { value: 300, name: 'Video Ads' }
       ],
       emphasis: {
         itemStyle: {
           shadowBlur: 10,
           shadowOffsetX: 0,
           shadowColor: 'rgba(0, 0, 0, 0.5)'
         }
       }
     }
   ]
 });
 
 
 // 更新数据的方法
 function updateChartData() {
   // 这里是新的数据，可以根据实际需要来设置
   option.value.series[0].data = [
     { value: Math.random() * 1000, name: 'Search Engine' },
     { value: Math.random() * 1000, name: 'Direct' },
     { value: Math.random() * 1000, name: 'Email' },
     // 添加或更新数据
   ];
 } 

 
// 定义柱状图配置作为响应式数据
const barOption = ref({
  title: {
    text: '月度销售额',
    left: 'center'
  },
  tooltip: {
    trigger: 'axis'
  },
  legend: {
    data: ['销售额'],
    top: '20px'
  },
  xAxis: {
    type: 'category',
    data: ['一月', '二月', '三月', '四月', '五月', '六月', '七月']
  },
  yAxis: {
    type: 'value',
    name: '销售额（元）'
  },
  series: [
    {
      name: '销售额',
      type: 'bar',
      data: [12000, 15000, 18000, 20000, 22000, 25000, 28000]
    }
  ]
});

// 更新柱状图数据的方法
function updateBarData() {
  const newData = ['一月', '二月', '三月', '四月', '五月', '六月', '七月'].map(() => {
    return Math.floor(Math.random() * 30000) + 5000;
  });
  barOption.value.series[0].data = newData;
} 
</script>

<style   scoped>
 /* 定义图表的大小 */
 .chart {
   width: 600px;
   height: 400px;
 }
 .box{
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
 }
 .update-chart-btn {
  border: 1px solid #1e40af;
  border-radius: 4px;
  padding: 8px 16px;
  background-color: #1e40af;
  color: white;
  cursor: pointer;
  transition: all 0.3s ease;
}
</style>