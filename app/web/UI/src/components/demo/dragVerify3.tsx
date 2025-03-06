import { defineComponent, ref } from 'vue'
import { ElCard } from 'element-plus'
import style from '@/assets/demo/dragVerify3.module.less'
export default defineComponent({
  setup() {
    const isDragging = ref(false)
    const isVerified = ref(false)

    const handleDragStart = (event: DragEvent) => {
      isDragging.value = true
      event.dataTransfer?.setData('text/plain', event.target.id)
    }

    const handleDragEnd = () => {
      isDragging.value = false
    }

    const handleDrop = (event: DragEvent) => {
      event.preventDefault()
      const data = event.dataTransfer?.getData('text/plain')
      const draggedElement = document.getElementById(data || '')
      if (draggedElement) {
        isVerified.value = true
      }
    }

    return () => (
      <div>
        <ElCard header="拖动验证" class={style.verify}>
          <div class="drag-container">
            <div id="drag-item" draggable onDragstart={handleDragStart} onDragend={handleDragEnd}>
              拖动我进行验证
            </div>
            <div id="drop-target" onDragover={(e) => e.preventDefault()} onDrop={handleDrop} />
          </div>
          {isVerified.value ? <p>验证通过！</p> : <p>请将拖动块拖动到指定区域进行验证。</p>}
        </ElCard>
      </div>
    )
  }
})
