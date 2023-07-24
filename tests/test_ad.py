import pytest

from newsnlp.ad import create_ad, Ad


@pytest.fixture
def ads_data():

    return [

        # An ad. This is made up.
        #<a href="http://leeram.today/proxy/click/2/663ca60d-0ad7-4575-9a87-738be1090cc2/" rel="nofollow" target="_blank">
        #   alt="free stuff" class="img_ad" width="356" border="0">
        #      <img src="https://tpc.googlesyndication.com/simgad/17762204761861913346?sqp=4sqPyQQrQikqJwhfEAEdAAC0QiABKAEwCTgDQPCTCUgAUAFYAWBfcAJ4AcUBLbKdPg&amp;rs=AOga4qmUk7OI9PLNISdX6oqM8P9ikqUvkg" alt="" class="img_ad" width="356" border="0">
        # </a>
        {
            # 'width': 356,
            # 'height': 296,
            'caption': 'Funded by: ',
            'alt': 'free stuff',
            'base_url': 'https://www.igfm.sn',
            'target_url': 'http://leeram.today/proxy/click/2/663ca60d-0ad7-4575-9a87-738be1090cc2/',
            'img_url': '../data/campaigns/igfm/2023/06/16872658732234.jpeg',
        },

        # <div class="ethical-ad">
        #     <a href="http://leeram.today/proxy/click/2/663ca60d-0ad7-4575-9a87-738be1090cc2/" rel="nofollow" target="_blank">
        #         <img src="https://storage.googleapis.com/leeram-ads/images/2023/06/sidebar-350x292.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&amp;X-Goog-Credential=docker-sa%40leeram.iam.gserviceaccount.com%2F20230707%2Fauto%2Fstorage%2Fgoog4_request&amp;X-Goog-Date=20230707T193954Z&amp;X-Goog-Expires=86400&amp;X-Goog-SignedHeaders=host&amp;X-Goog-Signature=51b87b4fe2b468067d8a438794ec48d3e0d6e0a1c15fbbd0f8d7da19050ee80ec83a863262d4101e055b42afe144353fcf661422c44e54c4ef8044601a6cac5256a1bcb784847600e4f5fefe696134bdeff219b99e6b18ff42d032d9e2fb6e58f182d9e91ab3c558cb252965628778a35bb8650ee2b052a127337429335a2161bced5c2ef75c9028230ed9dbc6aec1ab7ad7c98b21a8814eecb86c5ebe700ad7896ce8fe5b2616c12fcb64ec9699518d295f015574fb3dbf71dcaa9caeb1f33357d40678eec35b5eaee4e37042a3a768c27a5fcf403d9e80d9cdece07fd1cb44b7a46d9767bce71fb93a5ee23fe511e4b11df04ea294fa04ad4e3026912ebcd6">
        #     </a>
        #     <p class="ethical-text">
        #         <a href="http://leeram.today/proxy/click/2/663ca60d-0ad7-4575-9a87-738be1090cc2/" rel="nofollow noopener" target="_blank">
        #             Advertising space for rent. Image (350x292) with headline , content (100 chars) and call-to-action.
        #         </a>
        #     </p>
        # </div>
        {
            'caption': 'Advertising space for rent. Image (350x292) with headline , content (100 chars) and call-to-action.',
            'alt': None,
            'base_url': 'http://leeram.today',
            'target_url': 'http://leeram.today/proxy/click/2/663ca60d-0ad7-4575-9a87-738be1090cc2/',
            'img_url': 'https://storage.googleapis.com/leeram-ads/images/2023/06/sidebar-350x292.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&amp;X-Goog-Credential=docker-sa%40leeram.iam.gserviceaccount.com%2F20230707%2Fauto%2Fstorage%2Fgoog4_request&amp;X-Goog-Date=20230707T193954Z&amp;X-Goog-Expires=86400&amp;X-Goog-SignedHeaders=host&amp;X-Goog-Signature=51b87b4fe2b468067d8a438794ec48d3e0d6e0a1c15fbbd0f8d7da19050ee80ec83a863262d4101e055b42afe144353fcf661422c44e54c4ef8044601a6cac5256a1bcb784847600e4f5fefe696134bdeff219b99e6b18ff42d032d9e2fb6e58f182d9e91ab3c558cb252965628778a35bb8650ee2b052a127337429335a2161bced5c2ef75c9028230ed9dbc6aec1ab7ad7c98b21a8814eecb86c5ebe700ad7896ce8fe5b2616c12fcb64ec9699518d295f015574fb3dbf71dcaa9caeb1f33357d40678eec35b5eaee4e37042a3a768c27a5fcf403d9e80d9cdece07fd1cb44b7a46d9767bce71fb93a5ee23fe511e4b11df04ea294fa04ad4e3026912ebcd6'
        },

        # Ad by Google
        # <a  id="aw0" target="_top"
        #     href="https://googleads.g.doubleclick.net/aclk?sa=l&amp;ai=CQCj6l2CoZMCuJdfJtOUPkMK-2A7ryvTGcb2rocq4EaiBwbn2AxABIPP_4CRgrwWgAfSk6d4CyAECqAMByAPJBKoE_AFP0OVLWq-IKBgcw8GGNa5Afb2iKHo8p8I2B0jVPqYVujReqkWM5aqL8IS3c6P5THGWcH8iKvfJjAUVobFvFTuRtyvsyDhTlLtSvLyY8hL-UyhnbGsW0vO1cE2vdFm9Qdx1w8szE8wDpqSfBNz94fdN_dmA1lB8LSon9cy4UxXFnxAXrQh9fvgsJ10jQMMppIehzLrxtue2-T6RVDcVmLMqHuYb0A2Qk5YPaHNrsTRVFFmgT4sMTF-Z0OCiSQhbfsoQVxeubJIW2zWX0zVK8-4MWaRlqlsTFCFaPF3FK_91wnR3zgLvCnS3I6NwPMNEC4RV7Y92FP7MZooMlqPABLeesYW0BKAGAoAH9NqWoQGoB47OG6gHk9gbqAfulrECqAf-nrECqAeko7ECqAfVyRuoB6a-G6gHmgaoB_PRG6gHltgbqAeqm7ECqAeDrbECqAf_nrECqAffn7EC2AcB0ggUCIBhEAEYHzICigI6AoBASL39wTqxCT-JMOwwxE1DgAoBmAsByAsBuAwB2BMD0BUBmBYB-BYBgBcB&amp;ae=1&amp;num=1&amp;cid=CAQSTABpAlJWdYt_UW3xBTyB3HvTs0NrbKdb_cy-39sOY7ApGE4p3LnlpaIRhMizH00pKMl0LlqpdgpVZLnvCwvDs5WbgzRkbVqGRq-JyA8YAQ&amp;sig=AOD64_2EW6r3401R3bBcLXsaS421Y8PlTQ&amp;client=ca-pub-1061267829283799&amp;rf=2&amp;nb=17&amp;adurl=https://www.casamancaise.com/%3Fgclid%3DEAIaIQobChMIgNaF36P9_wIV1yStBh0QoQ_rEAEYASAAEgLDqfD_BwE" data-asoch-targets="ad0"><img src="https://tpc.googlesyndication.com/simgad/17762204761861913346?sqp=4sqPyQQrQikqJwhfEAEdAAC0QiABKAEwCTgDQPCTCUgAUAFYAWBfcAJ4AcUBLbKdPg&amp;rs=AOga4qmUk7OI9PLNISdX6oqM8P9ikqUvkg"
        #     alt="" class="img_ad" width="356" border="0">
        #         <img src="https://tpc.googlesyndication.com/simgad/17762204761861913346?sqp=4sqPyQQrQikqJwhfEAEdAAC0QiABKAEwCTgDQPCTCUgAUAFYAWBfcAJ4AcUBLbKdPg&amp;rs=AOga4qmUk7OI9PLNISdX6oqM8P9ikqUvkg" alt="" class="img_ad" width="356" border="0">
        # </a>
        {
            'caption': None,
            'alt': '',
            'base_url': 'https://igfm.sn',
            'target_url': 'https://googleads.g.doubleclick.net/aclk?sa=l&amp;ai=CQCj6l2CoZMCuJdfJtOUPkMK-2A7ryvTGcb2rocq4EaiBwbn2AxABIPP_4CRgrwWgAfSk6d4CyAECqAMByAPJBKoE_AFP0OVLWq-IKBgcw8GGNa5Afb2iKHo8p8I2B0jVPqYVujReqkWM5aqL8IS3c6P5THGWcH8iKvfJjAUVobFvFTuRtyvsyDhTlLtSvLyY8hL-UyhnbGsW0vO1cE2vdFm9Qdx1w8szE8wDpqSfBNz94fdN_dmA1lB8LSon9cy4UxXFnxAXrQh9fvgsJ10jQMMppIehzLrxtue2-T6RVDcVmLMqHuYb0A2Qk5YPaHNrsTRVFFmgT4sMTF-Z0OCiSQhbfsoQVxeubJIW2zWX0zVK8-4MWaRlqlsTFCFaPF3FK_91wnR3zgLvCnS3I6NwPMNEC4RV7Y92FP7MZooMlqPABLeesYW0BKAGAoAH9NqWoQGoB47OG6gHk9gbqAfulrECqAf-nrECqAeko7ECqAfVyRuoB6a-G6gHmgaoB_PRG6gHltgbqAeqm7ECqAeDrbECqAf_nrECqAffn7EC2AcB0ggUCIBhEAEYHzICigI6AoBASL39wTqxCT-JMOwwxE1DgAoBmAsByAsBuAwB2BMD0BUBmBYB-BYBgBcB&amp;ae=1&amp;num=1&amp;cid=CAQSTABpAlJWdYt_UW3xBTyB3HvTs0NrbKdb_cy-39sOY7ApGE4p3LnlpaIRhMizH00pKMl0LlqpdgpVZLnvCwvDs5WbgzRkbVqGRq-JyA8YAQ&amp;sig=AOD64_2EW6r3401R3bBcLXsaS421Y8PlTQ&amp;client=ca-pub-1061267829283799&amp;rf=2&amp;nb=17&amp;adurl=https://www.casamancaise.com/%3Fgclid%3DEAIaIQobChMIgNaF36P9_wIV1yStBh0QoQ_rEAEYASAAEgLDqfD_BwE" data-asoch-targets="ad0"><img src="https://tpc.googlesyndication.com/simgad/17762204761861913346?sqp=4sqPyQQrQikqJwhfEAEdAAC0QiABKAEwCTgDQPCTCUgAUAFYAWBfcAJ4AcUBLbKdPg&amp;rs=AOga4qmUk7OI9PLNISdX6oqM8P9ikqUvkg',
            'img_url': 'https://tpc.googlesyndication.com/simgad/17762204761861913346?sqp=4sqPyQQrQikqJwhfEAEdAAC0QiABKAEwCTgDQPCTCUgAUAFYAWBfcAJ4AcUBLbKdPg&amp;rs=AOga4qmUk7OI9PLNISdX6oqM8P9ikqUvkg',
        },

        # Not an ad
        # Search button on igfm.sn
        {
            'width': '20',
            'alt': 'search icon',
            'caption': ' RECHERCHE',
            'base_url': 'https://igfm.sn',
            'target_url': 'search.php',
            'img_url': 'theme/icons/search.png'
        },

        # <a href="https://www.facebook.com/CanalGuinee/">
        #     <img width="300" height="250"
        #         src="https://www.africaguinee.com/app/uploads/2023/05/Festival-Canal-v2.gif"
        #         class=" wp-post-image" alt="" loading="lazy" title="">
        # </a>
        {
            'width': '300',
            'height': '250',
            'alt': '',
            'caption': '',
            'base_url': 'https://igfm.sn',
            'target_url': 'https://www.facebook.com/CanalGuinee/',
            'img_url': 'https://www.africaguinee.com/app/uploads/2023/05/Festival-Canal-v2.gif'
        },
    ]


@pytest.fixture
def test_create_ad(ads_data):
    ad_candidate = ads_data[0]
    output = create_ad(lang='fr', **ad_candidate)
    assert output
    return ad_candidate


def test_ad(ad=test_create_ad):

    # has ad all fields?
    assert set(ad.dtypes) == set(map(lambda f: f.name, Ad.inputs))
