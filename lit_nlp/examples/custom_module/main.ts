/**
 * @fileoverview Main entrypoint for custom LIT demo.
 */

// Imports the main LIT App web component, which is declared here then attached
// to the DOM as <lit-app>
import '../../app/app';

import {app} from '../../core/lit_app';
import {LAYOUTS} from '../../default/layout';

import {ClassificationModule} from '../../modules/classification_module';
import {DataTableModule} from '../../modules/data_table_module';
import {DatapointEditorModule} from '../../modules/datapoint_editor_module';
import {PotatoModule} from './potato';

// Define a custom layout which includes our spud-tastic potato module!
LAYOUTS['potato'] = {
  components: {
    'Main': [DatapointEditorModule, ClassificationModule],
    'Data': [DataTableModule, PotatoModule],
  },
};

// Initialize the app core logic, using the specified declared layout.
app.initialize(LAYOUTS);
