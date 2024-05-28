import { ref,  defineComponent, type PropType } from 'vue'
import type{  ObjectType,Direction } from '../types'
import {
  ElCollapse,
  type CollapseModelValue,
  ElCollapseItem,
  ElDescriptions,
  ElDescriptionsItem
} from 'element-plus'

export interface Info {
  [key: string]: any
}
export interface Details {
  name: string
  data: Info
}

export default defineComponent({
  name: 'EcDetail',
  props: {
    data: {
      type: Array<Details>,
      required: true
    },
    column:{
      type: Number,
      default:3
    } 
    ,direction:{ 
      type: String as  PropType< Direction>,
      default:"vertical"
    }
  },
  emits: {
    activeChange: (activeIndex: CollapseModelValue, indexs: number[]) => true
  },

  setup(prop, context) {
    const activeNames = ref([0])
    const handleChange = (val: CollapseModelValue) => {
      context.emit('activeChange', val, activeNames.value)
    }
    return () => { 
      
      //可以写某些代码
      return (
        <>
          <div class="container">
            <ElCollapse v-model={activeNames.value} onChange={handleChange}>
              {prop.data.map((item, index) => {
                return (
                  <ElCollapseItem key={index} title={item.name} name={index}>
                    <div>
                      <ElDescriptions direction={prop.direction} column={prop.column} border>
                        {Object.keys(item.data).map((key) => (
                          <ElDescriptionsItem key={key} label={key}>
                            {item.data[key]}
                          </ElDescriptionsItem>
                        ))}
                      </ElDescriptions>
                    </div>
                  </ElCollapseItem>
                )
              })}
            </ElCollapse>
          </div>
        </>
      )
    }
  }
})
