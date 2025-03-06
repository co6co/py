import { defineComponent, ref } from 'vue'
import { ElSlider } from 'element-plus'

export default defineComponent({
  setup() {
    const sliderValue = ref(0)
    const isVerified = ref(false)

    const handleSliderChange = (value: number) => {
      if (value === 100) {
        isVerified.value = true
      } else {
        isVerified.value = false
      }
    }

    return () => (
      <div>
        <ElSlider vModel={sliderValue.value} min={0} max={100} onChange={handleSliderChange} />
        {isVerified.value ? <p>验证通过！</p> : <p>请将滑块拖动到最右侧进行验证。</p>}
      </div>
    )
  }
})
