import { type PropType, defineComponent, ref } from 'vue';
import { type ObjectType } from '../constants';
import { Detail, Details, Dialog } from '../components';
import type { DetailInstance, DialogInstance } from '../components';

export default defineComponent({
	name: 'DiaglogDetail',
	props: {
		title: {
			type: String,
		},
		column: {
			type: Number,
			default: 3,
		},
		data: {
			type: Object as PropType<Array<Details>>,
			required: true,
		},
	},
	setup(prop, ctx) {
		const dialogRef = ref<DialogInstance>();
		const openDiaLog = () => {
			if (dialogRef.value) {
				dialogRef.value.data.visible = true;
			}
		};
		const slots = {
			default: () => <Detail column={prop.column} data={prop.data}></Detail>,
			buttons: () => <> </>,
		};
		ctx.expose({
			openDiaLog,
		});
		const rander = (): ObjectType => {
			return (
				<Dialog
					title={prop.title}
					style={ctx.attrs}
					ref={dialogRef}
					v-slots={slots}></Dialog>
			);
		};
		rander.openDiaLog = openDiaLog;
		return rander;
	}, //end setup
});
