import { Logining } from './components/Logining' // 引入封装好的组件
import intervalTime from './components/intervalTime' // 但尚未设置 "--jsx"  设置tsconfig.compilerOptions{..., "jsx": "preserve",  ...}
import EcDetail from './components/common/EcDetail'

const components = [Logining, intervalTime,EcDetail]
const install = function (app: any, options: any) { 
  components.forEach((component) => {
    app.component(component.name, component)
  })
}
export default install // 批量的引入*
