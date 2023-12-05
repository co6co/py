<template>
    <div class="container"> 
        <div container="jess_player" ref="jess_player_container">

        </div> 
        {{player_option.type}}
        <el-checkbox-group v-model="player_option.type">
            <el-checkbox label="MediaSource" name="type"  >MediaSource</el-checkbox>
            <el-checkbox label="Webcodec " name="type" >Webcodec</el-checkbox>
            <el-checkbox label="SIMD" name="type" >SIMD </el-checkbox>
        </el-checkbox-group>
        <div class="input">
            <span>渲染标签：</span>
            <select id="renderDom" onchange="replay()">
                <option value="video" selected>video</option>
                <option value="canvas" >canvas</option>
            </select>

            <span>canvas渲染技术：</span>
            <select id="isUseWebGPU" onchange="replay()">
                <option value="webgl" >webgl</option>
                <option value="webgpu" selected>webgpu</option>
            </select>
            <span id="supportWebgpu"></span>
        </div>
        <!--
        <div class="input">
            <div>
                <span>缓存时长：</span>
                <input placeholder="单位：秒" type="text" id="videoBuffer" style="width: 50px" value="0.2">秒
                <span>缓存延迟(延迟超过会触发丢帧)：</span>
                <input placeholder="单位：秒" type="text" id="videoBufferDelay" style="width: 50px" value="1">秒
                <button id="replay">重播</button>
            </div>
        </div>
        -->
        <div class="input">
            <div>输入URL：</div>
            <input
                autocomplete="on"
                id="playUrl"
                value=""
            />
            <button id="play">播放</button>
            <button id="pause" style="display: none">停止</button>
        </div>
        <div class="input" style="line-height: 30px">
            <button id="destroy">销毁</button>
            <span class="fps-inner"></span>
        </div>
    </div>
</template>
 
<script setup lang="ts">
import { watch, PropType,reactive, ref , computed ,onMounted, onBeforeUnmount,nextTick} from 'vue';  
import "../../../assets/jessi/jessibuca-pro-demo.js";
import "../../../assets/jessi/jessibuca-pro-talk-demo.js";
import "../../../assets/jessi/demo.js";
import { PiniaVuePlugin } from 'pinia';


var $player = document.getElementById('play');
var $pause = document.getElementById('pause');
//var $playHref = document.getElementById('playUrl');
//var $container = document.getElementById('container');
var $destroy = document.getElementById('destroy');
/* var $useMSE = document.getElementById('useMSE');
var $useSIMD = document.getElementById('useSIMD');
var $useWCS = document.getElementById('useWCS');
*/
//var $videoBuffer = document.getElementById('videoBuffer');
//var $videoBufferDelay = document.getElementById('videoBufferDelay');
/*
var $replay = document.getElementById('replay');
var $fps = document.querySelector('.fps-inner');
var $renderDom = document.getElementById('renderDom');
var $isUseWebGPU = document.getElementById('isUseWebGPU');
*/
var showOperateBtns = true; // 是否显示按钮
var forceNoOffscreen = true; //
var jessibuca = null;

interface PlayerOption{
    type:Array<Boolean> 
    ,videoBuffer:number// 缓存时长 s
    ,videoBufferDelay:number// 缓存延迟 s
    ,useCanvasRender:boolean
    ,useWebGPU:boolean
} 
interface stream_data{
    url:string,
    quality:Array<String>,// ['普清', '高清', '超清', '4K', '8K']
}
const props = defineProps({
  data: {
    type:Array ,
    required: false
  } 
})
const player_option=ref<PlayerOption>({
    type:[true,true,true],
    videoBuffer:0.2,
    videoBufferDelay:2,
    useCanvasRender:false,
    useWebGPU:false 
})
const jess_player_container=ref<HTMLElement>( )
const jess_player=ref( )
const play=(url:string)=>{
    function play() { 
        if (url) {
            jess_player.value.play(url);
            if($player&&$pause&&$destroy){
                $player.style.display = 'none';
                $pause.style.display = 'inline-block';
                $destroy.style.display = 'inline-block';
            } 
        }
    } 
}
const create=()=>  { 
        console.log(jess_player_container.value)
       const jessibuca = new JessibucaPro({
            container:jess_player_container.value,
            videoBuffer: player_option.value.videoBuffer, // 缓存时长
            videoBufferDelay: player_option.value.videoBufferDelay, // 1000s
            isResize: false,
            text: "",
            loadingText: "加载中",
            debug: true,
            debugLevel: "debug",
            useMSE:player_option.value.type[0],// $useMSE.checked === true,
            useSIMD:player_option.value.type[1],// $useSIMD.checked === true,
            useWCS: player_option.value.type[2],//$useWCS.checked === true,
            /*
            showBandwidth: showOperateBtns, // 显示网速
            showPerformance: showOperateBtns, // 显示性能
            operateBtns: {
                fullscreen: showOperateBtns,
                screenshot: showOperateBtns,
                play: showOperateBtns,
                audio: showOperateBtns,
                ptz: showOperateBtns,
                quality: showOperateBtns,
                performance: showOperateBtns,
            },
            timeout: 10000,
            heartTimeoutReplayUseLastFrameShow: true,
            audioEngine: "worklet",
            qualityConfig: ['普清', '高清', '超清', '4K', '8K'],
            forceNoOffscreen: forceNoOffscreen,
            isNotMute: false,
            heartTimeout: 10,
            ptzZoomShow:true,
            useCanvasRender: player_option.value.useCanvasRender,
            useWebGPU: player_option.value.useWebGPU,
            controlHtml:'<div>我是 <span style="color: red">test</span>文案</div>',
            supportHls265: true,
            */
        },);


        jessibuca.on('ptz', (arrow:number) => {
            console.log('ptz', arrow);
        })

        jessibuca.on('streamQualityChange', (value:string) => {
            console.log('streamQualityChange', value);
        })

        jessibuca.on('timeUpdate', (value:string) => {
            console.log('timeUpdate', value);
        }) 
        if ($player&&$pause&&$destroy){ 
            $player.style.display = 'inline-block';
            $pause.style.display = 'none';
            $destroy.style.display = 'none'; 
        }
        jess_player.value=jessibuca
    } 
    onMounted(()=>{
        create()
    })
   
</script>
<style lang="less" scoped>
//@import "@/assets/jessi/demo.css";

.root {
    display: flex;
    place-content: center;
    margin-top: 3rem;
}

.container-shell {
    backdrop-filter: blur(5px);
    background: hsla(0, 0%, 50%, 0.5);
    padding: 30px 4px 10px 4px;
    /* border: 2px solid black; */
    width: auto;
    position: relative;
    border-radius: 5px;
    box-shadow: 0 10px 20px;
    &::before{
        content: "jessibuca demo player";
        position: absolute;
        color: darkgray;
        top: 4px;
        left: 10px;
        text-shadow: 1px 1px black;
    }
}

.container-shell:before {
    content: "jessibuca demo player";
    position: absolute;
    color: darkgray;
    top: 4px;
    left: 10px;
    text-shadow: 1px 1px black;
} 
 
.container {
    background: rgba(13, 14, 27, 0.7);
    width: 320px;
    height: 199px;
    display: inline-block;
    margin-right: 10px;
    margin-bottom: 10px;
} 
@media (max-width: 720px) { 
    .container {
        width: 95vw;
        height: 52.7vw;
        margin: 0 auto;
        margin-bottom: 10px;
        display: block;
    }
} 
</style>