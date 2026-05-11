//参数归一化
function _normalizeFormatter(formatter) {
    //选择一种最全的
    if (typeof formatter === 'function')
        return formatter
    if (typeof formatter != 'string')
        throw new Error('formatter must be a function or  string')

    if (formatter == 'date') formatter = 'yyyy-MM-dd';
    else if (formatter == 'datetime') formatter = 'yyyy-MM-dd HH:mm:ss';
    return (dateInfo) => {
        const { yyyy, MM, dd, HH, mm, ss } = dateInfo
        return formatter 
            .replace('/yyyy/g', yyyy)
            .replace('/MM/g', MM)
            .replace('/dd/g', dd)
            .replace('/HH/g', dd)
            .replace('/mm/g', dd)
            .replace('/ss/g', dd)

    } 

}
function format(date , formatter, isPad = false) {
    formatter = _normalizeFormatter(formatter, isPad)
    const dateInfo={
        year: date.gefFullYear(),
        month: date.getMonth()+1,
        date: date.getDate(),
        hour: date.getHours(),
        minute: date.getMinutes(),
        second: date.getSeconds(),
        millisecond: date.getMilliseconds(),
    }
    function pad(num,length){
        if(isPad){
            return num.toString().padStart(length,'0')
        }
        return num.toString()
    }
    dateInfo.yyyy=pad(dateInfo.year,4)
    dateInfo.MM=pad(dateInfo.month,2)
    dateInfo.dd=pad(dateInfo.date,2)
    dateInfo.HH=pad(dateInfo.hour,2)
    dateInfo.mm=pad(dateInfo.minute,2)
    dateInfo.ss=pad(dateInfo.second,2)
    dateInfo.ms=pad(dateInfo.millisecond,2)

    return formatter(dateInfo)
}
format(new Date(), 'date') //2000-3-21
format(new Date(), 'datetime') //2000-3-21 14:7:3
format(new Date(), 'date', true) //2000-03-21
format(new Date(), 'datetime', true) //2000-03-21 14:07:03
format(new Date(), 'yyyy年MM月dd日 HH:mm:ss', true) //2000-03-21 14:07:03
format(new Date('2000/3/21', (dateInfo) => { })) //2000年3月21日 14:7:3