# Copyright (C) 2017 DataArt
# Modifications copyright (C) 2018 lukerm
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by lukerm (2018). View the original file at:
#     https://github.com/devicehive/devicehive-audio-analysis/blob/master/log_config.py

LOGGING = {
    'version': 1,
    'formatters': {
        'simple': {
            'format': '[%(levelname)s] %(asctime)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'formatter': 'simple',
            'filename': 'logs/capture.log',
            'when': 'D',
            'interval': 1,
            'backupCount': 31,
        }
    },
    'loggers': {
        'audio_analysis': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False
        },
        'dh_webconfig': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False
        },
        'devicehive': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False
        },
    }
}
