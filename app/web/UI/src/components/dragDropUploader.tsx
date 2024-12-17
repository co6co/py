import { defineComponent, ref, reactive, onMounted, onBeforeUnmount, VNode } from 'vue'
import { Dialog, DialogInstance, showLoading, closeLoading, byte2Unit, IResponse } from 'co6co'

import { ElButton, ElMessageBox, type FormRules } from 'element-plus'
import { upload_svc, get_upload_chunks_svc } from '@/api/file'

export default defineComponent({
  name: 'ModifyTask',
  props: {
    title: {
      type: String
    },
    labelWidth: {
      type: Number, //as PropType<ObjectConstructor>,
      default: 110
    }
  },
  emits: {
    //@ts-ignore
    saved: (data: any) => true
  },
  setup(prop, ctx) {
    const diaglogForm = ref<DialogInstance>()
    const DATA = reactive<{ files: Array<File>; uploadFolder: String }>({
      files: [],
      uploadFolder: '/'
    })

    const readFileOrDirectory = (entry) => {
      //entry: FileEntry | DirectoryEntry
      if (entry.isFile) {
        entry.file((file: File) => {
          DATA.files.push(file)
        })
      } else if (entry.isDirectory) {
        const dirReader = entry.createReader()
        dirReader.readEntries((entries) => {
          entries.forEach((entry) => readFileOrDirectory(entry))
        })
      }
    }
    const onDragOver = (event) => {
      // 阻止默认行为，以允许文件被放置到目标区域
      event.preventDefault()
    }
    const onDrop = (event: DragEvent) => {
      const items = event.dataTransfer?.items
      if (items && items.length > 0) {
        for (let i = 0; i < items.length; i++) {
          //webkitGetAsEntry 是一个非标准的方法
          const item = items[i].webkitGetAsEntry()
          if (item) {
            readFileOrDirectory(item)
          } else {
            // 如果不是文件系统条目（可能是普通文件）
            const file = event.dataTransfer?.files[i]
            if (file) DATA.files.push(file)
          }
        }
      } else {
        // 如果 dataTransfer.items 为空，尝试从 dataTransfer.files 获取文件
        const files = event.dataTransfer?.files
        console.info('files', files)
        if (files && files.length > 0) {
          DATA.files = Array.from(files)
        }
      }
      // 阻止默认行为，以防止浏览器打开文件
      event.preventDefault()
    }
    const onUpload = async () => {
      if (!DATA.files || DATA.files.length == 0) {
        ElMessageBox.alert('请选择要上传的文件或文件夹！')
      }
      for (let i = 0; i < DATA.files.length; i++) {
        const file = DATA.files[i]
        const uploadedChunks = await getUploadedChunks(file.name)
        const chunks = createFileChunks(file)

        // 过滤掉已经上传的块
        const remainingChunks = chunks.filter((_, index) => !uploadedChunks.includes(index + 1))
        if (remainingChunks.length === 0) {
          console.log('所有块已上传完毕')
          continue
        }
        await uploadFileChunks(remainingChunks, file.name, chunks.length)
      }
    }
    /** 分片上传 */
    const createFileChunks = (file: File, chunkSize = 1 * 1024 * 1024) => {
      const chunks: Array<Blob> = []
      let start = 0
      while (start < file.size) {
        const end = Math.min(file.size, start + chunkSize)
        const chunk = file.slice(start, end)
        chunks.push(chunk)
        start = end
      }
      return chunks
    }
    const uploadFileChunks = async (chunks, fileName, totalChunks) => {
      for (let i = 0; i < chunks.length; i++) {
        const formData = new FormData()
        formData.append('file', chunks[i], `${fileName}_part${i + 1}`)
        formData.append('index', (i + 1).toString())
        formData.append('totalChunks', totalChunks)
        formData.append('fileName', fileName)

        try {
          const response = await upload_svc(DATA.uploadFolder + '/' + fileName, formData)
          console.log(`块 ${i + 1} 上传成功`, response.message)
        } catch (error) {
          console.error(`块 ${i + 1} 上传失败:`, error)
          // 处理错误，例如重试或提示用户
          break
        }
      }
    }
    const getUploadedChunks = async (fileName) => {
      const response = await get_upload_chunks_svc(`${DATA.uploadFolder}/${fileName}`)
      return response.data.uploadedChunks || []
    }

    /** end 分片上传 */
    const fromSlots = {
      buttons: () => (
        <>
          <ElButton type="warning" onClick={onUpload} v-slots={{ default: '开始上传' }} />
        </>
      ),
      default: () => (
        <>
          <div class="drop-zone" style="height:280px;" onDrop={onDrop} onDragover={onDragOver}>
            {DATA.files.length == 0 ? (
              <p>拖动文件或文件夹到这里</p>
            ) : (
              <ul>
                {DATA.files.map((file) => (
                  <li>
                    {file.name} {byte2Unit(file.size, 'b', 2)}
                    {new Date(file.lastModified).toISOString()}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </>
      )
    }

    const rander = (): VNode => {
      return (
        <Dialog
          closeTxt="取消上传"
          title={prop.title}
          style={ctx.attrs}
          ref={diaglogForm}
          v-slots={fromSlots}
        />
      )
    }
    const openDialog = (folder: string) => {
      DATA.uploadFolder = folder
      diaglogForm.value?.openDialog()
    }
    ctx.expose({
      openDialog
    })
    rander.openDialog = openDialog
    return rander
  } //end setup
})
