{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{
			"major" : 8,
			"minor" : 6,
			"revision" : 5,
			"architecture" : "x64",
			"modernui" : 1
		}
,
		"classnamespace" : "box",
		"rect" : [ 34.0, 171.0, 1029.0, 495.0 ],
		"bglocked" : 0,
		"openinpresentation" : 0,
		"default_fontsize" : 12.0,
		"default_fontface" : 0,
		"default_fontname" : "Arial",
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"gridsnaponopen" : 1,
		"objectsnaponopen" : 1,
		"statusbarvisible" : 2,
		"toolbarvisible" : 1,
		"lefttoolbarpinned" : 0,
		"toptoolbarpinned" : 0,
		"righttoolbarpinned" : 0,
		"bottomtoolbarpinned" : 0,
		"toolbars_unpinned_last_save" : 0,
		"tallnewobj" : 0,
		"boxanimatetime" : 200,
		"enablehscroll" : 1,
		"enablevscroll" : 1,
		"devicewidth" : 0.0,
		"description" : "",
		"digest" : "",
		"tags" : "",
		"style" : "",
		"subpatcher_template" : "",
		"showontab" : 1,
		"assistshowspatchername" : 0,
		"boxes" : [ 			{
				"box" : 				{
					"id" : "obj-4",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 0,
					"patching_rect" : [ 505.0, 354.0, 134.0, 22.0 ],
					"text" : "test_lstm_max_vs_msp"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-3",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 0,
					"patching_rect" : [ 505.0, 265.0, 109.0, 22.0 ],
					"text" : "rtneural_lstm_pred"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-2",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 0,
					"patching_rect" : [ 505.0, 183.0, 113.0, 22.0 ],
					"text" : "rtneural_mlp_adder"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-1",
					"maxclass" : "newobj",
					"numinlets" : 0,
					"numoutlets" : 0,
					"patching_rect" : [ 505.0, 99.0, 118.0, 22.0 ],
					"text" : "rtneural_mlp_control"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-8",
					"linecount" : 7,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 58.0, 60.0, 363.0, 100.0 ],
					"text" : "rtneural is the control rate max rtneural plugin object.\n\nthis object can load an neural network model saved in rtneural format for inference.\n\nif the patchers on the right are not available, be sure to add the rtneural_max folder to your path"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-7",
					"linecount" : 20,
					"maxclass" : "comment",
					"numinlets" : 1,
					"numoutlets" : 0,
					"patching_rect" : [ 486.0, 60.0, 489.0, 275.0 ],
					"text" : "use a multi layered perceptron to control a synthesizer\n    see the mlp/rtneural_mlp_control tutorial\n\n\n\n\nteach a multi layered perceptron to add\n    see the mlp/rtneural_mlp_adder tutorial\n\n\n\n\nteach a recurrent neural network about note prediction\n    see the lstm_note_prediction/rtneural_lstm_pred tutorial\n\n\n\n\nsee how to do note prediction at audio rate\n    see the test_lstm_max_vs_msp tutorial"
				}

			}
 ],
		"lines" : [  ],
		"parameters" : 		{
			"obj-4::obj-32" : [ "level", "level", 0 ],
			"parameterbanks" : 			{
				"0" : 				{
					"index" : 0,
					"name" : "",
					"parameters" : [ "-", "-", "-", "-", "-", "-", "-", "-" ]
				}

			}
,
			"inherited_shortname" : 1
		}
,
		"dependency_cache" : [ 			{
				"name" : "div_by_maxval.js",
				"bootpath" : "~/Documents/Max 8/Library/rtneural_max/examples/lstm_note_prediciton",
				"patcherrelativepath" : "./lstm_note_prediciton",
				"type" : "TEXT",
				"implicit" : 1
			}
, 			{
				"name" : "get_lstm_vectors.js",
				"bootpath" : "~/Documents/Max 8/Library/rtneural_max/examples/lstm_note_prediciton",
				"patcherrelativepath" : "./lstm_note_prediciton",
				"type" : "TEXT",
				"implicit" : 1
			}
, 			{
				"name" : "get_note_durs.js",
				"bootpath" : "~/Documents/Max 8/Library/rtneural_max/examples/lstm_note_prediciton",
				"patcherrelativepath" : "./lstm_note_prediciton",
				"type" : "TEXT",
				"implicit" : 1
			}
, 			{
				"name" : "rtneural.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "rtneural_lstm_pred.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/rtneural_max/examples/lstm_note_prediciton",
				"patcherrelativepath" : "./lstm_note_prediciton",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "rtneural_mlp_adder.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/rtneural_max/examples/mlp",
				"patcherrelativepath" : "./mlp",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "rtneural_mlp_control.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/rtneural_max/examples/mlp",
				"patcherrelativepath" : "./mlp",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "rtneural~.mxo",
				"type" : "iLaX"
			}
, 			{
				"name" : "rtneural~_mlp_trigger_mode.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/rtneural_max/examples/mlp",
				"patcherrelativepath" : "./mlp",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "test_lstm_max_vs_msp.maxpat",
				"bootpath" : "~/Documents/Max 8/Library/rtneural_max/examples/lstm_note_prediciton",
				"patcherrelativepath" : "./lstm_note_prediciton",
				"type" : "JSON",
				"implicit" : 1
			}
, 			{
				"name" : "top_n_weighted.js",
				"bootpath" : "~/Documents/Max 8/Library/rtneural_max/examples/lstm_note_prediciton",
				"patcherrelativepath" : "./lstm_note_prediciton",
				"type" : "TEXT",
				"implicit" : 1
			}
 ],
		"autosave" : 0
	}

}
