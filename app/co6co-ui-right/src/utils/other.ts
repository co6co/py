import { IEnumSelect } from 'co6co';


export const getTagTypeStr = (list: IEnumSelect[], checked: (item: IEnumSelect) => boolean) => {
    const s = [undefined, "success", "danger", "info", "warning", "primary"]
    let index = list?.find(checked)?.value
    if (index === undefined) return undefined
    index = Number(index)
    if (index < 0 || index >= s.length) return undefined
    return s[index]
}