
<template>
    <!--一开始测试使用-->
    <div class="box">   
        <div class="jess_player" ref="jess_player_container">  </div>   
        <!--
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
         {{  `FPS: ${fps.fps} DFPS: ${fps.dfps}` }}  {{ player_option.useWebGPU }}  {{ player_option.currentSource?.name }}
        --> 
    </div>
</template>
 
<script setup lang="ts">
import { watch, PropType,reactive, ref , computed ,onMounted, onUnmounted, onBeforeUnmount,nextTick, watchEffect} from 'vue';  

    //import '../../../assets/jessi/jessibuca-pro.js';
	//import '../../../assets/jessi/jessibuca-pro-talk.js';
    //import '../../../assets/jessi/demo.js'; 
  
    import '../../../assets/jessi/jessibuca-pro-demo.js';
	import '../../../assets/jessi/jessibuca-pro-talk-demo.js';
   // import '../../../assets/jessi/demo.js'; 
	
 
var showOperateBtns = true; // 是否显示按钮 
const props = defineProps({
    sources: {
    type: Array<stream_source>   ,
    required: true
  } ,
  option: {
    type:Object as  PropType<player_option> ,
    required: false
  } 
})
 
const fps=reactive({fps:0,dfps:0}) 
const player_option=reactive<PlayerOption>({
    type:"MediaSource",
    renderDom:"video",
    useWebGPU:false,
    videoBuffer:0.2,
    videoBufferDelay:2,
    useCanvasRender:false,  
})
let currentSource:stream_source|undefined=undefined
watchEffect(() => {
    try {
      
        if (props.sources.length>0 && props.sources[0].url) {
            currentSource=props.sources.at(0)
            nextTick(()=>{ replay()})   // 可能会死
        }else{

        }

    } catch (e) {
        console.error(e)
    }
})
 
const jess_player_container=ref<HTMLElement>( )
const jess_player=ref()  
const emits=defineEmits(["created","destroyed"])
 
const onPlay=( )=>{   
    try{ 
        if (currentSource )
            jess_player.value.play(currentSource.url)  
        
    }catch(e){
        console.error(e)
    } 
} 
const destroying=()=>{
    emits("destroyed")
}
const replay=()=> {   
    try{ 
        if (jess_player.value) { 
            jess_player.value.destroy().then(() =>{ 
                destroying(),create(),  onPlay()  
            });  
        } 
        else {
            create(), onPlay();
        }  
    }catch(e){
        console.error(e)
    } 
}

/*
const check_paler=()=> new Promise((resolve, reject) => {
        if (JessibucaPro) {
            resolve(null)
        } else { 
            loadJs('./jessibuca-pro-demo.js', 'jessibuca-pro', function () {
                resolve()
            })
        }
}) 
check_paler().then(() => {
    create();
 })

*/ 
const create=()=>  {  
    console.info("create") 
    const jessibuca = new JessibucaPro({
        container:jess_player_container.value,
        videoBuffer: player_option.videoBuffer, // 缓存时长
        videoBufferDelay: player_option.videoBufferDelay, // 1000s 
        isResize: false,
        text: "text",
        loadingText: "加载中",
        debug: false,
        debugLevel: "debug",
        useMSE:player_option.type=="MediaSource",// $useMSE.checked === true,
        useSIMD:player_option.type=="Webcodec",// $useSIMD.checked === true,
        useWCS: player_option.type=="SIMD",//$useWCS.checked === true,
        
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
        useCanvasRender: player_option.renderDom=="canvas",
        useWebGPU: player_option.useWebGPU,
        //controlHtml:'<div>我是 <span style="color: red">test</span>文案</div>',
        supportHls265: true,
        qualityConfig: props.sources.map(m=>m.name),
    },); 
    jessibuca.on('ptz', (arrow:number) => {
        console.log('ptz', arrow);
    })

    jessibuca.on('streamQualityChange', (value:string) => {
        //todo 需要调试功能 
        currentSource=props.sources.find(m=>m.name==value)
        //console.log('streamQualityChange', value,player_option.currentSource,"all=>",props.sources);
        onPlay()  
    })

    jessibuca.on('timeUpdate', (value:string) => { 
    }) 
    jessibuca.on('stats', (stats:{fps:number,dfps:number}) => { 
        fps.dfps=stats .dfps
        fps.fps=stats.fps 
    })  
    jess_player.value=jessibuca
    emits("created",jessibuca)
} 
onMounted(()=>{ 
    //create(),onPlay()
}) 
onUnmounted(()=>{
    if (jess_player.value) jess_player.value.destroy().then(() =>destroying()); 
})
const onReplay=()=>{
    replay()
}
const onPause=()=>{
    jess_player.value.pause();
}
</script>
<style lang="less" scoped> 
.jess_player{
    width:100%;
    height:500px;
    backdrop-filter: blur(5px);
    background: hsla(0, 0%, 50%, 0.5);
    padding: 0px 4px 0px 4px;
} 
</style>../../../assets/jessi--/jessibuca-pro-demo.js../../../assets/jessi--/jessibuca-pro-talk-demo.js../../../assets/jessi--/demo.js