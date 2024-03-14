import { ref, reactive, defineComponent, type PropType, inject, provide, computed ,onMounted,onUnmounted} from 'vue'

import { ElCard, ElMenu, ElMenuItem } from 'element-plus'
import type { ObjectType } from './types'

export interface ContextMenuItem {
  id: number | string
  name: number | string
}
export interface IContextMenu {
  visible: boolean
  left: number
  top: number
  data: ContextMenuItem[]
}
export default defineComponent({
  name: 'EcForm',
  emits: {
    checked: (index: number, item: ContextMenuItem) => true
  },
  setup(prop, { attrs, slots, emit, expose }) {
    const menuData = reactive<IContextMenu>({
      visible: false,
      left: 0,
      top: 0,
      data: []
    })
    const open = (data: ContextMenuItem[], event: MouseEvent) => {
      event.preventDefault() //阻止鼠标右键默认行为
      menuData.data = data
      menuData.left = event.clientX
      menuData.top = event.pageY
      menuData.visible = true
      console.info('ddddddd', data)
    }

    const onSelectMenu = (index: number, item: ContextMenuItem) => {
      menuData.visible = false
      emit('checked', index, item)
    }
    const style = computed(() => {
      return {
        left: menuData.left + 'px',
        top: menuData.top + 'px',
        position: 'fixed',
        'z-index': 9
      }
    })
    const close=()=>{
      menuData.visible = false
    }
    onMounted(()=>{
      document.addEventListener("click",close)
    })
    onUnmounted(()=>{
      document.removeEventListener("click",close)
    })
    const render = (): ObjectType => {
      //可以写某些代码
      return (
        <>
          {menuData.visible ? (
            <div style={style.value}> 
                <ElMenu mode="vertical" style="">
                  {menuData.data.map((item, index) => {
                    return (
                      <ElMenuItem
                        style="height:32px;line-height:32px;border-bottom: 1px solid var(--el-border-color);"
                        index={index.toString()}
                        onClick={() => onSelectMenu(index, item)}
                      >
                        {item.name}
                      </ElMenuItem>
                    )
                  })}
                </ElMenu> 
            </div>
          ) : (
            <></>
          )}
        </>
      )
    }

    expose({
      open
    })
    render.open = open

    return render
  }
})
