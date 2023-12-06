<template>
    <div class="box">  
        <div class="jess_player" ref="jess_player_container">  </div>  
        <el-radio-group v-model="player_option.type" @change="onReplay">
            <el-radio label="MediaSource" />
            <el-radio label="Webcodec" />
            <el-radio label="SIMD" />
        </el-radio-group>

        <el-radio-group v-model="player_option.renderDom" @change="onReplay">
            <el-radio label="video" />
            <el-radio label="canvas" /> 
        </el-radio-group>
        <el-radio-group v-model="player_option.useWebGPU" @change="onReplay">  
            <el-radio :label="true"  >使用webGPU</el-radio>
            <el-radio :label="false" >不使用webGPU</el-radio>
        </el-radio-group>  
        {{  `FPS: ${fps.fps} DFPS: ${fps.dfps}` }}  {{ player_option.useWebGPU }}  
    </div>
</template>
 
<script setup lang="ts">
import { watch, PropType,reactive, ref , computed ,onMounted, onBeforeUnmount,nextTick} from 'vue';  
import "../../../assets/jessi/jessibuca-pro-demo.js";
import "../../../assets/jessi/jessibuca-pro-talk-demo.js";
import "../../../assets/jessi/demo.js";  
  
var showOperateBtns = true; // 是否显示按钮 
const props = defineProps({
  sources: {
    type: Array<stream_source> ,
    required: true
  } ,
  option: {
    type:Object as  PropType<player_option> ,
    required: false
  } 
})
const fps=ref({fps:0,dfps:0}) 
const player_option=ref<PlayerOption>({
    type:"MediaSource",
    renderDom:"video",
    useWebGPU:false,
    videoBuffer:0.2,
    videoBufferDelay:2,
    useCanvasRender:false, 
    currentSource:-1, 
})
const jess_player_container=ref<HTMLElement>( )
const jess_player=ref() 
const emits=defineEmits(["created","destroyed"])
const onPlay=( )=>{ 
    let index=player_option.value.currentSource
    let url=undefined;
    if (index>-1 && index < props.sources.length ) url=props.sources.at(index)?.url 
    if (url) jess_player.value.play(url);  
} 
const destroying=()=>{
    emits("destroyed")
}
const replay=()=> {
    alert("配置改变重新播放")
    if (jess_player.value) {
        jess_player.value.destroy().then(() =>destroying(),create(),onPlay());
    } else {
        create(),onPlay();
    }
}
const create=()=>  {  
       const jessibuca = new JessibucaPro({
            container:jess_player_container.value,
            videoBuffer: player_option.value.videoBuffer, // 缓存时长
            videoBufferDelay: player_option.value.videoBufferDelay, // 1000s
            isResize: false,
            text: "text",
            loadingText: "加载中",
            debug: false,
            debugLevel: "debug",
            useMSE:player_option.value.type=="MediaSource",// $useMSE.checked === true,
            useSIMD:player_option.value.type=="Webcodec",// $useSIMD.checked === true,
            useWCS: player_option.value.type=="SIMD",//$useWCS.checked === true,
            
            showBandwidth: showOperateBtns, // 显示网速
            //showPerformance: showOperateBtns, // 显示性能
            operateBtns: {
                fullscreen: showOperateBtns,
                screenshot: showOperateBtns,
                play: showOperateBtns,
                audio: showOperateBtns,
                //ptz: showOperateBtns,
                quality: showOperateBtns,
                performance: showOperateBtns,
            },
            
            timeout: 10000,
            heartTimeoutReplayUseLastFrameShow: true,
            audioEngine: "worklet",
          
            forceNoOffscreen: true, 
            isNotMute: false,
            heartTimeout: 10,
            ptzZoomShow:true,
            useCanvasRender: player_option.value.renderDom=="canvas",
            useWebGPU: player_option.value.useWebGPU,
            //controlHtml:'<div>我是 <span style="color: red">test</span>文案</div>',
            supportHls265: true,
            qualityConfig:props.sources.map(m=>m.name),
        },);


        jessibuca.on('ptz', (arrow:number) => {
            console.log('ptz', arrow);
        })

        jessibuca.on('streamQualityChange', (value:string) => {
            player_option.value.currentSource=props.sources.findIndex(m=>m.name==value)
            console.log('streamQualityChange', value,player_option.value.currentSource);
        })

        jessibuca.on('timeUpdate', (value:string) => {
           // console.log('timeUpdate', value);
        }) 
        jessibuca.on('stats', (stats:{fps:number,dfps:number}) => {
            // console.log('stats', stats);
            fps.value=stats 
        })
        
        jess_player.value=jessibuca
        emits("created",jessibuca)
    } 
      onMounted(()=>{
        create()
        onPlay()
    }) 
    const onReplay=()=>{
        replay()
    }
    const onPause=()=>{
        jess_player.value.pause();
    }
</script>
<style lang="less" scoped>
//@import "@/assets/jessi/demo.css";
.jess_player{
    width:100%;
    height:400px;
    backdrop-filter: blur(5px);
    background: hsla(0, 0%, 50%, 0.5);
    padding: 30px 4px 10px 4px;
} 
</style>