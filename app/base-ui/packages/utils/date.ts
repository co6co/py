//https://www.cnblogs.com/gaosj20210301/p/17348910.html

/*
  timestamp: 13位时间戳 | new Date() | Date()
  console.log(dateFormat(1714528800000, 'YY-MM-DD HH:mm:ss'))
  format => YY：年，M：月，D：日，H：时，m：分钟，s：秒，SSS：毫秒
*/
export const format = (
  timestamp: number | string | Date,
  format = 'YYYY-MM-DD HH:mm:ss'
) => {
  const date = new Date(timestamp)
  function fixedTwo(value: number): string {
    return value < 10 ? `0${value}` : String(value)
  }
  let showTime = format
  if (showTime.includes('SSS')) {
    const S = date.getMilliseconds()
    showTime = showTime.replace('SSS', '0'.repeat(3 - String(S).length) + S)
  }
  if (showTime.includes('YY')) {
    const Y = date.getFullYear()
    showTime = showTime.includes('YYYY')
      ? showTime.replace('YYYY', String(Y))
      : showTime.replace('YY', String(Y).slice(2, 4))
  }
  if (showTime.includes('M')) {
    const M = date.getMonth() + 1
    showTime = showTime.includes('MM')
      ? showTime.replace('MM', fixedTwo(M))
      : showTime.replace('M', String(M))
  }
  if (showTime.includes('D')) {
    const D = date.getDate()
    showTime = showTime.includes('DD')
      ? showTime.replace('DD', fixedTwo(D))
      : showTime.replace('D', String(D))
  }
  if (showTime.includes('H')) {
    const H = date.getHours()
    showTime = showTime.includes('HH')
      ? showTime.replace('HH', fixedTwo(H))
      : showTime.replace('H', String(H))
  }
  if (showTime.includes('m')) {
    const m = date.getMinutes()
    showTime = showTime.includes('mm')
      ? showTime.replace('mm', fixedTwo(m))
      : showTime.replace('m', String(m))
  }
  if (showTime.includes('s')) {
    const s = date.getSeconds()
    showTime = showTime.includes('ss')
      ? showTime.replace('ss', fixedTwo(s))
      : showTime.replace('s', String(s))
  }
  return showTime
}
/*
interface IdateForamtObject{
  [key: string]: number,

}

export const format = (date: Date, fmt: string) => {
  if (!fmt) return ''
  const re: RegExp = /(y+)/
  if (re.test(fmt)) {
    const data=re.exec(fmt)
    if(data ){
      const t = data[1] 
      fmt = fmt.replace(t, (date.getFullYear() + '').substring(4 - t.length))
    }
  }

  const o:IdateForamtObject = { 
    'M+': date.getMonth() + 1, // 月份
    'd+': date.getDate(), // 日
    'h+': date.getHours(), // 小时
    'm+': date.getMinutes(), // 分
    's+': date.getSeconds(), // 秒
    'q+': Math.floor((date.getMonth() + 3) / 3), // 季度
     S: date.getMilliseconds() // 毫秒
  } 
  for (const k in o) {
    const regx = new RegExp('(' + k + ')')
    if (regx.test(fmt)) {
      const data=regx.exec(fmt)
      if(data ){
        const t = data[1]   
        fmt = fmt.replace(t,t.length == 1 ?  o[k]   : ('00' + o[k]).substr(('' + o[k]).length))
      }
      
    }
  }
  return fmt
}
*/
