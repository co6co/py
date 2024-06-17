import { ref, defineComponent, type PropType, watch } from 'vue'

import { ElImage, ElEmpty,ElIcon } from 'element-plus'
import {  Picture} from '@element-plus/icons-vue';  
import imgStyle from '../../assets/css/images.module.less' 
export default defineComponent({
  name: 'ImagesList',
  props: {
    list: {
      //请求的url会加上 baseURL
      type: Object as PropType<string[]>,
      required: true
    }
  }, 
  setup(props) {
    const currentImageUrl = ref<string>() 
    const loading = ref<boolean>() 
    watch(
      () => props.list,
      (n: string[]) => {
        if (n && n.length > 0) {
          currentImageUrl.value = n[0]
        }
      }
    ) 
    return () => {
      //可以写某些代码
      return (
        <>   
        <div class={imgStyle.imageList}>   
          {currentImageUrl.value ? (
            <ElImage  class={imgStyle.imagesList}
             
              src={currentImageUrl.value}
              style="width: 100%; height: 100%"
              zoom-rate={1.2}
              max-scale={7}
              min-scale={0.2}
              fit="cover"
              preview-src-list={props.list}
              v-slots={{ 
                error:()=>{
                  return (<div class={imgStyle.image_slot}> <ElIcon><Picture/></ElIcon></div>) 
                } 
              }}
            />  
          ) : (
            <ElEmpty description="未加载数据" />
          )} 
        </div>
        </>
      )
    }
  }
})  