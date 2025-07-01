import { rm } from 'fs/promises'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

async function cleanViteCache() {
  try {
    const cachePath = path.join(__dirname, 'node_modules', '.vite', 'deps')
    await rm(cachePath, { recursive: true, force: true })
    console.log('✅ Vite缓存已清理')
  } catch (error) {
    console.error('❌ 清理缓存时出错:', error)
  }
}

cleanViteCache()
