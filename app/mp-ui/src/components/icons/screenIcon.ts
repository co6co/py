import { defineComponent, createElementVNode, openBlock, createElementBlock } from 'vue'
interface svgEle {
  [key: string]: any
}
interface svgPath extends svgEle {
  d: string
  stroke?: string //="rgba(6, 148, 56, 1)"
  'stroke-width'?: number
  fill?: string
}
interface svgRect extends svgEle {
  stroke?: string //="rgba(6, 148, 56, 1)"
  x?: string
  y?: string
  width?: string
  height?: string
  rx?: string
}
interface svgEleKey {
  [key: string]: any
}
type dOrEle = svgEleKey | string
const _createElementBlock = (dOrPathArr: dOrEle[], xml: any = undefined) => {
  let vnode = dOrPathArr.map((d) => {
    if (typeof d == 'string') {
      return createElementVNode('path', {
        fill: 'currentColor',
        d: d
      })
    } else {
      let ds = Object.keys(d) //只能有一个key 
      return createElementVNode(ds[0], d[ds[0]])
    }
  })
  if (!xml) {
    xml = {
      xmlns: 'http://www.w3.org/2000/svg',
      viewBox: '0 0 1024 1024'
    }
  }
  return createElementBlock('svg', xml, vnode)
}
export const OneScreen = defineComponent({
  name: 'OneScreen',
  setup(__props) {
    const d =
      'M894.44 118.63H129.56C93.35 118.63 64 147.99 64 184.2v655.6c0 36.21 29.35 65.56 65.56 65.56h764.88c36.21 0 65.56-29.35 65.56-65.56V184.2c0-36.21-29.35-65.57-65.56-65.57z m21.85 721.17H107.71V227.9H916.3v611.9z'
    return (ctx: any, cache: any) => (openBlock(), _createElementBlock([d]))
  }
})
export const TwoScreen = defineComponent({
  name: 'TwoScreen',
  setup(__props) {
    const d = [
      'M900.5 130.1v763.8h-777V130.1h777m48-60h-873c-6.6 0-12 5.4-12 12v859.8c0 6.6 5.4 12 12 12h873c6.6 0 12-5.4 12-12V82.1c0-6.6-5.4-12-12-12z',
      'M482 126.5h60v771h-60z'
    ]
    return (ctx: any, cache: any) => (openBlock(), _createElementBlock(d))
  }
})
export const FourScreen = defineComponent({
  name: 'FourScreen',
  setup(__props) {
    const d =
      'M894.44 118.63H129.56C93.35 118.63 64 147.99 64 184.2v655.6c0 36.21 29.35 65.56 65.56 65.56H894.45c36.21 0 65.56-29.35 65.56-65.56V184.2c-0.01-36.21-29.36-65.57-65.57-65.57zM107.71 227.9h382.44V512H107.71V227.9z m0 611.9V555.7h382.44v284.1H107.71z m808.58 0H533.85V555.7h382.44v284.1z m0-327.8H533.85V227.9h382.44V512z'
    return (ctx: any, cache: any) => (openBlock(), _createElementBlock([d]))
  }
})
export const CanelFullScreen = defineComponent({
  name: 'CanelFullScreen',
  setup(__props) {
    const d =
      'M238.206348 284.981132h-190.665343c-9.660377 0-18.303873 4.067527-23.896723 10.168818-10.677259 10.677259-15.253227 28.981132-2.542205 44.23436 7.118173 8.643496 18.303873 13.727905 29.998014 13.727904h233.882821c37.624628 0 68.131082-30.506455 68.131082-68.131082v-237.441907c0-9.660377-4.067527-18.303873-10.168818-23.896723-10.677259-10.677259-28.981132-15.253227-44.23436-2.542204-8.643496 7.118173-13.727905 18.303873-13.727904 29.998014v187.106256l-228.289971-228.28997c-12.711023-13.219464-33.5571-13.219464-46.776564 0-12.711023 12.711023-12.711023 34.065541 0 46.776564l228.289971 228.28997zM284.982912 664.278054h-237.441907c-9.660377 0-18.303873 4.067527-23.896723 10.168818-10.677259 10.677259-15.253227 28.981132-2.542205 44.234359 7.118173 8.643496 18.303873 13.727905 29.998014 13.727905h187.106257l-228.289971 228.28997c-13.219464 12.711023-13.219464 33.5571 0 46.776564 12.711023 12.711023 34.065541 12.711023 46.776564 0l228.289971-228.28997v190.665343c0 9.660377 4.067527 18.303873 10.168818 23.896723 10.677259 10.677259 28.981132 15.253227 44.234359 2.542204 8.643496-7.118173 13.727905-18.303873 13.727905-29.998014v-233.88282c0.508441-37.624628-29.998014-68.131082-68.131082-68.131082zM1016.120945 967.308838l-228.28997-228.28997h190.665343c9.660377 0 18.303873-4.067527 23.896723-10.168818 10.677259-10.677259 15.253227-28.981132 2.542204-44.23436-7.118173-8.643496-18.303873-13.727905-29.998014-13.727904h-233.88282c-37.624628 0-68.131082 30.506455-68.131082 68.131082V976.460775c0 9.660377 4.067527 18.303873 10.168818 23.896723 10.677259 10.677259 28.981132 15.253227 44.234359 2.542204 8.643496-7.118173 13.727905-18.303873 13.727905-29.998014v-187.106256l228.28997 228.28997c12.711023 13.219464 33.5571 13.219464 46.776564 0 12.711023-12.711023 12.711023-34.065541 0-46.776564zM733.936238 353.112214h237.441907c9.660377 0 18.303873-4.067527 23.896723-10.168818 10.677259-10.677259 15.253227-28.981132 2.542204-44.234359-7.118173-8.643496-18.303873-13.727905-29.998013-13.727905h-187.106257l228.289971-228.28997c13.219464-12.711023 13.219464-33.5571 0-46.776564-12.711023-12.711023-34.065541-12.711023-46.776564 0l-228.289971 228.28997v-190.665343c0-9.660377-4.067527-18.303873-10.168818-23.896723-10.677259-10.677259-28.981132-15.253227-44.234359-2.542204-8.643496 7.118173-13.727905 18.303873-13.727905 29.998014v233.88282c0.508441 37.624628 31.014896 68.131082 68.131082 68.131082z'
    return (ctx: any, cache: any) => (openBlock(), _createElementBlock([d]))
  }
})

export const FullScreen = defineComponent({
  name: 'FullScreen',
  setup(__props) {
    const d =
      'M895.872 96.512c0 0.256 0.128 0.448 0.128 0.64l0 253.696C896 369.216 881.6 384.192 864 384c-17.6 0.064-32.064-14.72-32.064-33.216L831.936 173.312 631.232 374.016c-12.928 12.992-33.664 13.44-46.08 0.832C572.672 362.368 572.992 341.76 585.984 328.704L786.752 128 609.152 128C590.848 128 575.872 113.6 576 96 576 78.4 590.72 64 609.28 64l253.632 0c0.192 0 0.384 0.064 0.576 0.128C863.616 64.064 863.744 64 863.936 64c7.232 0 13.632 2.944 19.008 7.168 1.216 0.896 2.56 1.472 3.584 2.496 0.32 0.32 0.448 0.704 0.768 1.024C892.544 80.384 896 87.744 896 95.936 896 96.128 895.872 96.32 895.872 96.512zM173.248 128l177.6 0C369.152 128 384.128 113.6 384 96 384 78.4 369.28 64 350.72 64L97.152 64C96.96 64 96.768 64.064 96.576 64.128 96.384 64.064 96.256 64 96.064 64 88.832 64 82.432 66.944 77.056 71.168c-1.152 0.896-2.56 1.472-3.584 2.496C73.216 73.984 73.088 74.432 72.768 74.688 67.456 80.384 64 87.744 64 95.936c0 0.192 0.128 0.384 0.128 0.576C64.128 96.768 64 96.96 64 97.216l0 253.696C64 369.216 78.4 384.192 96 384c17.6 0.064 32.064-14.72 32.064-33.216L128.064 173.312l200.704 200.704c12.928 12.992 33.664 13.44 46.08 0.832 12.48-12.416 12.16-33.088-0.832-46.144L173.248 128zM896 609.152C896 590.784 881.6 575.808 864 576c-17.6-0.064-32.064 14.72-32.064 33.216l0 177.472L631.232 585.984c-12.928-12.992-33.664-13.44-46.08-0.832C572.672 597.632 572.992 618.24 585.984 631.296L786.752 832 609.152 832C590.848 832 575.872 846.4 576 864 576 881.6 590.72 896 609.28 896l253.632 0c0.192 0 0.384-0.128 0.576-0.128S863.744 896 863.936 896c7.232 0 13.568-2.944 18.944-7.168 1.216-0.896 2.624-1.408 3.648-2.496 0.32-0.32 0.448-0.704 0.768-1.024C892.544 879.616 896 872.32 896 864.064c0-0.192-0.128-0.384-0.128-0.64 0-0.192 0.128-0.384 0.128-0.64L896 609.152zM350.848 832 173.248 832l200.768-200.704C387.008 618.24 387.328 597.632 374.784 585.216c-12.352-12.608-33.152-12.16-46.08 0.832l-200.704 200.704L128 609.216C128.064 590.72 113.6 575.936 96 576 78.4 575.808 64 590.784 64 609.152l0 253.696c0 0.256 0.128 0.448 0.128 0.64 0 0.256-0.128 0.448-0.128 0.64 0 8.256 3.456 15.552 8.768 21.248 0.256 0.32 0.384 0.704 0.704 1.024 1.024 1.088 2.432 1.6 3.584 2.496C82.496 893.056 88.832 896 96.064 896c0.192 0 0.32-0.128 0.512-0.128S96.96 896 97.152 896L350.72 896C369.28 896 384 881.6 384 864 384.128 846.4 369.152 832 350.848 832z'
    return (ctx: any, cache: any) => (openBlock(), _createElementBlock([d]))
  }
})

//
const xml = {
  xmlns: 'http://www.w3.org/2000/svg',
  width: '47.60377502441406',
  height: '28',
  viewBox: '0 0 47.60377502441406 28',
   fill: 'none'
}
export const GreenVideoCamera = defineComponent({
  name: 'GreenVideoCamera',
  setup(__props) {
    const d = [
      {
        path: {
           fill: '#069438',
          d: 'M1 3.4375L1 24.5625C1 25.9087 2.09131 27 3.4375 27L36.2983 27C37.6445 27 38.7358 25.9087 38.7358 24.5625L38.7358 3.4375C38.7358 2.09131 37.6445 1 36.2983 1L3.4375 1C2.09131 1 1 2.09131 1 3.4375Z'
        }
      },
      {
        path: {
           fill: 'rgba(6, 148, 56, 1)',
          d: 'M3.4375 -1.19209e-07L36.2983 -1.19209e-07C38.1968 -1.19209e-07 39.7358 1.53902 39.7358 3.4375L39.7358 24.5625C39.7358 26.461 38.1968 28 36.2983 28L3.4375 28C1.53902 28 0 26.461 0 24.5625L0 3.4375C0 1.53902 1.53902 -1.19209e-07 3.4375 -1.19209e-07L3.4375 2C2.64359 2 2 2.64359 2 3.4375L2 24.5625C2 25.3564 2.64359 26 3.4375 26L36.2983 26C37.0923 26 37.7358 25.3564 37.7358 24.5625L37.7358 3.4375C37.7358 2.64359 37.0922 2 36.2983 2L3.4375 2L3.4375 -1.19209e-07Z'
        }
      },
      {
        path: {
           fill: 'rgba(6, 148, 56, 1)',
          d: 'M3.4375 -1.19209e-07L36.2983 -1.19209e-07C38.1968 -1.19209e-07 39.7358 1.53902 39.7358 3.4375L39.7358 24.5625C39.7358 26.461 38.1968 28 36.2983 28L3.4375 28C1.53902 28 0 26.461 0 24.5625L0 3.4375C0 1.53902 1.53902 -1.19209e-07 3.4375 -1.19209e-07L3.4375 2C2.64359 2 2 2.64359 2 3.4375L2 24.5625C2 25.3564 2.64359 26 3.4375 26L36.2983 26C37.0923 26 37.7358 25.3564 37.7358 24.5625L37.7358 3.4375C37.7358 2.64359 37.0922 2 36.2983 2L3.4375 2L3.4375 -1.19209e-07Z'
        }
      },
      {
        path: {
          d: 'M40 21.6875L46.6038 24.9375L46.6038 3L40 6.25L40 21.6875Z',
           stroke: 'rgba(6, 148, 56, 1)',
          'stroke-width': 2,
          fill: '#069438'
        }
      },
      {
        rect: {
          x: '5.717010498046875',
          y: '3.4375',
          width: '10.377365112304688',
          height: '4',
          rx: '2',
          fill: '#FFFFFF'
        }
      }
    ]

    return (ctx: any, cache: any) => (openBlock(), _createElementBlock(d, xml))
  }
})

export const RedVideoCamera = defineComponent({
  name: 'RedVideoCamera',
  setup(__props) {
    const d = [
      {
        path: {
          fill: 'rgba(255, 19, 3, 1)',
          d: 'M3.4375 -1.19209e-07L36.2983 -1.19209e-07C38.1968 -1.19209e-07 39.7358 1.53902 39.7358 3.4375L39.7358 24.5625C39.7358 26.461 38.1968 28 36.2983 28L3.4375 28C1.53902 28 0 26.461 0 24.5625L0 3.4375C0 1.53902 1.53902 -1.19209e-07 3.4375 -1.19209e-07L3.4375 2C2.64359 2 2 2.64359 2 3.4375L2 24.5625C2 25.3564 2.64359 26 3.4375 26L36.2983 26C37.0923 26 37.7358 25.3564 37.7358 24.5625L37.7358 3.4375C37.7358 2.64359 37.0922 2 36.2983 2L3.4375 2L3.4375 -1.19209e-07Z'
        }
      },
      {
        path: {
          d: 'M39 21.6875L45.6038 24.9375L45.6038 3L39 6.25L39 21.6875Z',
          stroke: 'rgba(255, 19, 3, 1)',
          'stroke-width': '2'
        }
      }, 
      {
        rect: {
          x: '5.717010498046875',
          y: '3.4375',
          width: '10.377365112304688',
          height: '4',
          rx: '2',
          fill: '#FF1303'
        }
      }
    ] 
    return (ctx: any, cache: any) => (openBlock(), _createElementBlock(d, xml))
  }
})