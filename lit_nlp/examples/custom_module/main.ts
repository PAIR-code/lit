/**
 * Test frontend.
 */

// Imports the main LIT App web component, which is declared here then attached
// to the DOM as <lit-app>
import '../../client/app/app';

import {app} from '../../client/core/lit_app';
import {LAYOUTS} from '../../client/default/layout';

import {ClassificationModule} from '../../client/modules/classification_module';
import {DataTableModule} from '../../client/modules/data_table_module';
import {DatapointEditorModule} from '../../client/modules/datapoint_editor_module';
import {PotatoModule} from './potato';

LAYOUTS['potato'] = {
  components: {
    'Main': [DatapointEditorModule, ClassificationModule],
    'Data': [DataTableModule, PotatoModule],
  },
};

// Initialize the app core logic, using the specified declared layout.
app.initialize(LAYOUTS);

