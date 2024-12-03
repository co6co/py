import { defineComponent, ref, reactive, provide, onMounted, VNode } from 'vue'
import { ElConfigProvider } from 'element-plus'
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import enUs from 'element-plus/es/locale/lang/en'
import './assets/css/main.css'
import './assets/css/color-dark.css'
export default defineComponent({
  name: 'APP',
  setup(props, ctx) {
    const data = ref<{ theme: string; locale: any; size: '' | 'default' | 'small' | 'large' }>({
      theme: 'dark',
      size: '',
      locale: zhCn
    })
    const changeLang = (locale: 'en' | 'zh') => {
      if (locale === 'en') {
        data.value.locale = enUs
      } else if (locale === 'zh') {
        data.value.locale = zhCn
      }
    }
    const rander = (): VNode => {
      return (
        <ElConfigProvider locale={data.value.locale}>
          <Router-View />
        </ElConfigProvider>
      )
    }

    //ctx.expose({ })
    //rander.openDialog = openDialog
    return rander
  }
})
