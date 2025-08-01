import { type PropType, defineComponent, ref, VNode, reactive, computed } from 'vue';
import { Dialog, DialogInstance ,getBaseUrl} from "co6co"
import { ImageView, ImageViewInstance, image2Option } from "co6co-right" 

export default defineComponent({
    name: 'DiaglogDetail',
    props: { 
        path: {
            type: String,
            required: true,
        },
    },
    setup(prop, ctx) { 
        const dialogRef = ref<DialogInstance>();
        const openDialog = (title:string) => {
            titleRef.value=title
            if (dialogRef.value) {
                dialogRef.value.data.visible = true;
            }
        };
         const titleRef=ref("")
        const imgOption=computed(()=>{
            return {
                url: `${getBaseUrl()}/api/dev/img/imgpreview?path=${prop.path}`,
                authon: true
            }
        })
        const slots = {
            default: () => <ImageView option={imgOption.value}></ImageView>,
            buttons: () => <> </>,
        };
        ctx.expose({
            openDialog,
        });
        const rander = (): VNode => {
            return (
                <Dialog
                    title={titleRef.value}
                    style={ctx.attrs}
                    ref={dialogRef}
                    v-slots={slots}></Dialog>
            );
        };
        rander.openDialog = openDialog;
        return rander;
    }, //end setup
});
