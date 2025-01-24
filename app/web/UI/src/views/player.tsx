import { computed, defineComponent, nextTick, onUnmounted, VNodeChild, watch } from 'vue'
import { ref, reactive, onMounted } from 'vue'
import { ElButton, ElContainer, ElRow, ElCol, ElInput } from 'element-plus'
import { getPublicURL } from '@/utils'
import { loadScript, unLoadScript } from 'co6co'
export default defineComponent({
  setup(_, __) {
    const DATA = reactive<{
      url: string
    }>({
      url: 'https://1500005692.vod2.myqcloud.com/43843706vodtranscq1500005692/62656d94387702300542496289/v.f100240.m3u8'
    })
    //:use

    //const playerJsPath = getPublicURL('/player/hls.0.12.4_hava.js')
    //const playerJsPath = 'https://web.sdk.qcloud.com/player/tcplayerlite/release/v2.4.5/TcPlayer-2.4.5.js'
    const playerJsPath = getPublicURL('/player/TcPlayer-2.3.2_hava.js')
    onMounted(() => {
      loadScript(playerJsPath, () => {
        createPlayer()
      })
    })
    onUnmounted(() => {
      // 如果需要在组件卸载前移除脚本标签（可选）
      unLoadScript(playerJsPath)
    })
    const playerRef = ref()

    const createPlayer = () => {
      var player = new TcPlayer('id_test_video', {
        m3u8: DATA.url, //请替换成实际可用的播放地址
        m3u8_hd: 'http://200002949.vod.myqcloud.com/200002949_b6ffc.f230.av.m3u8',
        m3u8_sd: 'http://200002949.vod.myqcloud.com/200002949_b6ffc.f220.av.m3u8',
        autoplay: true, //iOS 下 safari 浏览器，以及大部分移动端浏览器是不开放视频自动播放这个能力的
        poster: 'http://www.test.com/myimage.jpg',
        width: '480', //视频的显示宽度，请尽量使用视频分辨率宽度
        height: '320' //视频的显示高度，请尽量使用视频分辨率高度
      })
      playerRef.value = player
      console.info(player)
    }
    const onPlayer = () => {
      playerRef.value.destroy()
      createPlayer()
    }
    const onStop = () => {
      playerRef.value.stop()
    }
    //:page reader
    const rander = (): VNodeChild => {
      return (
        <ElContainer>
          <ElRow>
            <ElCol>
              <ElInput v-model={DATA.url}>
                {{
                  append: () => (
                    <>
                      <ElButton onClick={onPlayer}>播放</ElButton>
                      <ElButton onClick={onStop}>停止</ElButton>
                    </>
                  )
                }}
              </ElInput>
              <div id="id_test_video" style="width:100%; height:auto;"></div>
            </ElCol>
          </ElRow>
        </ElContainer>
      )
    }
    return rander
  } //end setup
})
