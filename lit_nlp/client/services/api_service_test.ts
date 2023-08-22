import 'jasmine';

import {mockMetadata, mockSerializedMetadata} from '../lib/testing_utils';
import {StatusService} from './status_service';
import {ApiService} from './api_service';

const STATUS_SERVICE = new StatusService();

describe('api service test', () => {
  let apiService: ApiService;
  beforeEach(async () => {
    apiService = new ApiService(STATUS_SERVICE);
  });

  it('correctly deserializes LitMetadata from /get_info', async () => {
    const getInfoResponse = new Response(
        JSON.stringify(mockSerializedMetadata),
        {status: 200, statusText: 'OK'}
    );
    spyOn(window, 'fetch').and.resolveTo(getInfoResponse);
    const metadata = await apiService.getInfo();
    expect(window.fetch).toHaveBeenCalledWith('./get_info?', {
      method: 'POST', body: JSON.stringify({inputs: []})
    });
    expect(metadata).toEqual(mockMetadata);
  });
});
